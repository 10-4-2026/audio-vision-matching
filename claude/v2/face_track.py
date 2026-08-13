"""
face_track.py
--------------
Phát hiện và theo dõi khuôn mặt trong video, chọn ra track của TÀI XẾ
(dựa trên vị trí khuôn mặt trong khung hình - có thể chỉnh theo camera thực tế),
sau đó cắt vùng miệng (mouth ROI) cho từng frame để phục vụ bước Active
Speaker Detection (ASD) ở file asd.py.

Dùng thư viện `uniface` (RetinaFace) để detect + landmark 5 điểm
(mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải).
"""

from dataclasses import dataclass, field
import cv2
import numpy as np
from uniface import RetinaFace


@dataclass
class FaceDetection:
    frame_idx: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    landmarks: np.ndarray     # shape (5, 2)
    score: float


@dataclass
class FaceTrack:
    track_id: int
    detections: list = field(default_factory=list)  # list[FaceDetection]

    def mouth_center(self, det: FaceDetection):
        # landmark index 3,4 = khóe miệng trái/phải
        return det.landmarks[3:5].mean(axis=0)


class DriverFaceTracker:
    """
    Theo dõi khuôn mặt đơn giản bằng IoU-matching giữa các frame liên tiếp
    (đủ dùng cho video quay trong cabin xe, ít khuôn mặt, camera cố định).
    """

    def __init__(self, det_thresh: float = 0.6, iou_thresh: float = 0.3):
        self.detector = RetinaFace(model_name="retinaface_mnet_v2", conf_thresh=det_thresh)
        self.iou_thresh = iou_thresh
        self.tracks: dict[int, FaceTrack] = {}
        self._next_id = 0

    @staticmethod
    def _iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _match_track(self, bbox):
        best_id, best_iou = None, self.iou_thresh
        for tid, track in self.tracks.items():
            if not track.detections:
                continue
            last_bbox = track.detections[-1].bbox
            iou = self._iou(bbox, last_bbox)
            if iou > best_iou:
                best_id, best_iou = tid, iou
        return best_id

    def run(self, video_path: str, sample_stride: int = 1) -> dict:
        """
        Chạy detect + track qua toàn bộ video.
        sample_stride: xử lý 1 frame mỗi N frame để tăng tốc (ASD vẫn cần
        khớp timeline gốc nên nên để =1 nếu máy đủ mạnh).
        Trả về dict {track_id: FaceTrack}.
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_stride == 0:
                boxes, landmarks = self.detector.detect(frame)  # API của uniface
                for bbox, lmk, score in self._iter_dets(boxes, landmarks):
                    tid = self._match_track(bbox)
                    if tid is None:
                        tid = self._next_id
                        self._next_id += 1
                        self.tracks[tid] = FaceTrack(track_id=tid)
                    det = FaceDetection(frame_idx=frame_idx, bbox=bbox,
                                         landmarks=lmk, score=score)
                    self.tracks[tid].detections.append(det)
            frame_idx += 1
        cap.release()
        return self.tracks

    @staticmethod
    def _iter_dets(boxes, landmarks):
        """
        Chuẩn hoá output của uniface về dạng (bbox[4], landmarks[5,2], score).
        Tuỳ version uniface, boxes có thể là Nx5 ([x1,y1,x2,y2,score]).
        """
        for i in range(len(boxes)):
            b = boxes[i]
            bbox = np.array(b[:4], dtype=np.float32)
            score = float(b[4]) if len(b) > 4 else 1.0
            lmk = np.array(landmarks[i], dtype=np.float32).reshape(5, 2)
            yield bbox, lmk, score


def select_driver_track(tracks: dict, frame_width: int, driver_side: str = "left") -> FaceTrack:
    """
    Heuristic chọn track của tài xế dựa trên VỊ TRÍ khuôn mặt trong khung hình.
    - driver_side="left": tài xế ở nửa trái khung hình (camera đặt giữa xe nhìn ra,
      hoặc camera hành trình quay ngược vào cabin - cần chỉnh theo thực tế lắp đặt).
    - Ngoài ra ưu tiên track có nhiều detection nhất (xuất hiện liên tục, ổn định).

    Nếu bố trí camera của bạn khác (vd: driver luôn là khuôn mặt lớn nhất /
    gần camera nhất), hãy thay hàm này bằng logic phù hợp, hoặc cho người
    dùng click chọn 1 lần ở frame đầu rồi track theo track_id đó.
    """
    candidates = []
    for tid, track in tracks.items():
        if len(track.detections) < 5:
            continue
        centers_x = [ (d.bbox[0] + d.bbox[2]) / 2 for d in track.detections ]
        avg_x = float(np.mean(centers_x))
        in_region = (avg_x < frame_width / 2) if driver_side == "left" else (avg_x >= frame_width / 2)
        candidates.append((tid, track, in_region, len(track.detections)))

    # ưu tiên: đúng vùng vị trí -> nhiều detection nhất
    candidates.sort(key=lambda c: (not c[2], -c[3]))
    if not candidates:
        raise RuntimeError("Không tìm thấy track khuôn mặt nào đủ dài để coi là tài xế.")
    return candidates[0][1]


def crop_mouth(frame: np.ndarray, det: FaceDetection, size: int = 112) -> np.ndarray:
    """Cắt & resize vùng miệng quanh khóe miệng trái/phải, dùng cho model ASD."""
    mouth_c = det.landmarks[3:5].mean(axis=0)
    face_w = det.bbox[2] - det.bbox[0]
    half = face_w * 0.35
    x1, y1 = int(mouth_c[0] - half), int(mouth_c[1] - half * 0.7)
    x2, y2 = int(mouth_c[0] + half), int(mouth_c[1] + half * 0.7)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size))
