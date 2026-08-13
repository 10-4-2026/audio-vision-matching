"""
mouth_detector.py
Phát hiện khuôn mặt tài xế (chỉ có 1 người trong khung hình) và trích xuất
tín hiệu chuyển động khẩu hình (Mouth Aspect Ratio - MAR) theo thời gian.

Khác với bài toán "ai đang nhìn camera", ở đây tài xế KHÔNG cần nhìn thẳng
vào camera, nên module này không tính head pose/gaze mà chỉ tập trung vào
chuyển động môi (mở/đóng miệng) bất kể hướng đầu.

Dùng mediapipe 1.0.0 (Tasks API - FaceLandmarker). Model .task được tự
động tải về ở lần chạy đầu tiên.
"""
from __future__ import annotations
import os
import sys
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"
)

# Landmark index (topology 468 điểm chuẩn của mediapipe face mesh)
# dùng để tính Mouth Aspect Ratio (MAR).
_UPPER_INNER_LIP = 13
_LOWER_INNER_LIP = 14
_LEFT_MOUTH_CORNER = 61
_RIGHT_MOUTH_CORNER = 291
# Thêm 1 cặp điểm phụ (mép trong môi 2 bên) để làm mượt phép đo độ mở miệng
_UPPER_INNER_LIP_2 = 82
_LOWER_INNER_LIP_2 = 87


def _ensure_model(model_path: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[mouth_detector] Không tìm thấy model tại {model_path}, đang tải về...",
          file=sys.stderr)
    try:
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("[mouth_detector] Tải model thành công.", file=sys.stderr)
    except Exception as e:
        raise RuntimeError(
            f"Không tải được model FaceLandmarker tự động ({e}).\n"
            f"Vui lòng tải thủ công tại:\n  {_MODEL_URL}\n"
            f"rồi đặt vào:\n  {model_path}\n"
            f"(hoặc truyền model_path trỏ tới file bạn đã tải)."
        ) from e
    return model_path


class MouthSignalExtractor:
    """
    Trích xuất Mouth Aspect Ratio (MAR) của khuôn mặt DUY NHẤT trong khung
    hình (tài xế) qua từng frame. num_faces=1 vì chỉ có tài xế trong video.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        model_path: str | None = None,
    ):
        model_path = _ensure_model(model_path or _DEFAULT_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,  # chỉ có tài xế trong khung hình
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _mar_from_landmarks(pts, w: int, h: int) -> tuple[float, tuple]:
        def P(idx):
            return np.array([pts[idx].x * w, pts[idx].y * h])

        v1 = np.linalg.norm(P(_UPPER_INNER_LIP) - P(_LOWER_INNER_LIP))
        v2 = np.linalg.norm(P(_UPPER_INNER_LIP_2) - P(_LOWER_INNER_LIP_2))
        horiz = np.linalg.norm(P(_LEFT_MOUTH_CORNER) - P(_RIGHT_MOUTH_CORNER))
        horiz = max(horiz, 1e-6)
        mar = ((v1 + v2) / 2.0) / horiz

        xs = [pt.x * w for pt in pts]
        ys = [pt.y * h for pt in pts]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return float(mar), bbox

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        """
        Trả về dict: {'detected': bool, 'mar': float|None, 'bbox': tuple|None}
        """
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return {"detected": False, "mar": None, "bbox": None}

        pts = result.face_landmarks[0]  # chỉ có 1 khuôn mặt (num_faces=1)
        mar, bbox = self._mar_from_landmarks(pts, w, h)
        return {"detected": True, "mar": mar, "bbox": bbox}


def extract_mouth_signal(
    video_path: str,
    sample_stride: int = 1,
    model_path: str | None = None,
    smooth_window: int = 5,
) -> dict:
    """
    Xử lý toàn bộ video, trả về tín hiệu MAR theo thời gian:
    {
        'times': np.ndarray (giây),
        'mar_raw': np.ndarray,
        'mar_smooth': np.ndarray,
        'detected_ratio': float (tỉ lệ frame phát hiện được mặt),
        'fps': float,
    }

    Frame không phát hiện được mặt sẽ được nội suy tuyến tính từ các frame
    lân cận (forward/backward fill + interpolate) để tín hiệu liên tục.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps or 25.0

    times, mar_values, detected_flags = [], [], []
    frame_idx = 0

    with MouthSignalExtractor(model_path=model_path) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_stride == 0:
                t_sec = frame_idx / fps
                res = extractor.process_frame(frame, int(t_sec * 1000))
                times.append(t_sec)
                detected_flags.append(res["detected"])
                mar_values.append(res["mar"] if res["detected"] else np.nan)
            frame_idx += 1

    cap.release()

    times = np.array(times, dtype=np.float64)
    mar_values = np.array(mar_values, dtype=np.float64)
    detected_flags = np.array(detected_flags, dtype=bool)
    detected_ratio = float(detected_flags.mean()) if len(detected_flags) else 0.0

    mar_filled = _interpolate_nan(mar_values)
    mar_smooth = _moving_average(mar_filled, smooth_window)

    return {
        "times": times,
        "mar_raw": mar_filled,
        "mar_smooth": mar_smooth,
        "detected_ratio": detected_ratio,
        "fps": float(fps),
    }


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    nans = np.isnan(x)
    if nans.all():
        return np.zeros_like(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")
