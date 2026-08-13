"""
asd.py
------
Active Speaker Detection (ASD): thay vì "đọc khẩu hình ra chữ" (rất khó,
gần như không khả thi cho tiếng Việt), ta chỉ dùng khẩu hình để trả lời câu
hỏi ĐƠN GIẢN HƠN NHIỀU: "tại thời điểm t, tài xế có đang nói hay không?"
bằng cách so khớp chuyển động môi với waveform âm thanh (audio-visual sync).

Khuyến nghị dùng model Light-ASD (nhẹ, nhanh, đủ chính xác cho video 1-3 người):
  repo: https://github.com/Junhua-Liao/Light-ASD
Hoặc TalkNet-ASD nếu cần độ chính xác cao hơn (chậm hơn):
  repo: https://github.com/TaoRuijie/TalkNet-ASD

File này định nghĩa interface chuẩn để cắm 1 trong 2 model trên vào pipeline.
Bạn cần tự tải checkpoint pretrained (.pth) của model đã chọn về máy, vì môi
trường này không có sẵn quyền truy cập mạng để tải checkpoint ngoài các domain
đã whitelist (pypi/github source code thì tải được, nhưng file weight lớn
thường host ở Google Drive/Dropbox nên cần bạn tải thủ công).
"""

from dataclasses import dataclass
import numpy as np
import torch
import librosa


@dataclass
class ASDResult:
    frame_idx: int
    speaking_score: float  # càng cao càng chắc chắn đang nói (thường 0..1 sau sigmoid)


class LightASDWrapper:
    """
    Wrapper gọi model Light-ASD. Cần:
      pip install torch torchaudio
      git clone https://github.com/Junhua-Liao/Light-ASD
    rồi trỏ checkpoint_path tới file weight .pth đã tải.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # TODO: import đúng class model từ repo Light-ASD sau khi clone, ví dụ:
        # from Light_ASD.model.Model import ASD_Model
        # self.model = ASD_Model()
        # self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        # self.model.to(device).eval()
        raise NotImplementedError(
            "Hãy clone repo Light-ASD, import đúng class model, load checkpoint, "
            "rồi bỏ dòng raise này. Interface .predict() bên dưới giữ nguyên."
        )

    @torch.no_grad()
    def predict(self, mouth_crops: np.ndarray, audio_mfcc: np.ndarray) -> np.ndarray:
        """
        mouth_crops: (T, 112, 112, 3) uint8 - chuỗi crop miệng đã align theo frame video
        audio_mfcc:  (T, n_mfcc) - đặc trưng âm thanh tương ứng cùng độ dài T
        return: (T,) điểm số "đang nói" cho mỗi frame, đã sigmoid về [0,1]
        """
        raise NotImplementedError


def extract_audio_features(wav: np.ndarray, sr: int, video_fps: float, n_mfcc: int = 13) -> np.ndarray:
    """
    Trích MFCC và resample về đúng số frame video (để khớp 1-1 với mouth_crops).
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    n_video_frames = int(len(wav) / sr * video_fps)
    mfcc_resampled = librosa.util.fix_length(mfcc, size=n_video_frames, axis=1)
    return mfcc_resampled.T  # (T, n_mfcc)


def run_asd(mouth_crops: np.ndarray, wav: np.ndarray, sr: int, video_fps: float,
            model: LightASDWrapper) -> np.ndarray:
    """Hàm tiện ích gộp bước trích audio feature + gọi model ASD."""
    audio_feat = extract_audio_features(wav, sr, video_fps)
    T = min(len(mouth_crops), len(audio_feat))
    return model.predict(mouth_crops[:T], audio_feat[:T])


def scores_to_segments(scores: np.ndarray, fps: float, threshold: float = 0.5,
                        min_duration_sec: float = 0.3, merge_gap_sec: float = 0.25):
    """
    Chuyển chuỗi điểm số speaking_score theo frame -> danh sách đoạn thời gian
    (start_sec, end_sec) mà tài xế đang nói. Có làm mượt: bỏ đoạn quá ngắn,
    và nối các đoạn cách nhau 1 khoảng ngắn (tránh bị cắt vụn do nhiễu).
    """
    speaking = scores >= threshold
    segments = []
    start = None
    for i, is_speak in enumerate(speaking):
        if is_speak and start is None:
            start = i
        elif not is_speak and start is not None:
            segments.append((start / fps, i / fps))
            start = None
    if start is not None:
        segments.append((start / fps, len(speaking) / fps))

    # lọc đoạn quá ngắn
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration_sec]

    # nối các đoạn gần nhau
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_sec:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged
