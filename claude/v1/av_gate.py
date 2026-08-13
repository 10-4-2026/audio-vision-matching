"""
av_gate.py
Cô lập giọng nói tài xế khỏi âm thanh hỗn tạp (hành khách nói chuyện, tạp
âm...) bằng cách tương quan (correlate) tín hiệu chuyển động khẩu hình
(từ video) với năng lượng âm thanh theo thời gian.

Nguyên lý: khi tài xế thực sự đang nói, chuyển động môi (MAR) và năng
lượng âm thanh sẽ biến thiên đồng bộ với nhau. Khi âm thanh đến từ nguồn
khác (hành khách, nhạc), MAR của tài xế sẽ không tương quan với năng
lượng âm thanh tại thời điểm đó. Ta dùng hệ số tương quan cục bộ (cửa sổ
trượt) để tạo ra 1 "gain mask" theo thời gian, rồi áp lên audio để giữ lại
đoạn tài xế nói và giảm/tắt các đoạn không liên quan.

Đây là phương pháp heuristic nhẹ (không cần model deep-learning riêng),
phù hợp làm MVP. Độ chính xác sẽ thấp hơn các model audio-visual speech
separation chuyên dụng (VisualVoice, Looking-to-Listen...) nhưng không đòi
hỏi GPU/model pretrained lớn.
"""
from __future__ import annotations
import numpy as np


def compute_audio_envelope(
    samples: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính năng lượng RMS ngắn hạn (short-time energy envelope) của audio.
    Trả về (times_sec, envelope).
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    n = len(samples)
    if n < frame_len:
        return np.array([0.0]), np.array([float(np.sqrt(np.mean(samples ** 2) + 1e-12))])

    n_frames = 1 + (n - frame_len) // hop_len
    env = np.empty(n_frames, dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        seg = samples[start:start + frame_len]
        env[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        times[i] = (start + frame_len / 2.0) / sr

    return times, env


def resample_to_grid(src_times: np.ndarray, src_values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính tín hiệu src (times,values) sang các mốc target_times."""
    if len(src_times) < 2:
        return np.full_like(target_times, src_values[0] if len(src_values) else 0.0, dtype=np.float64)
    return np.interp(target_times, src_times, src_values, left=src_values[0], right=src_values[-1])


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def windowed_correlation(
    sig_a: np.ndarray, sig_b: np.ndarray, sr_grid: float,
    window_sec: float = 1.0, hop_sec: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính hệ số tương quan Pearson cục bộ giữa 2 tín hiệu đã cùng lưới thời
    gian (sampling rate sr_grid mẫu/giây), theo cửa sổ trượt.
    Trả về (center_times_sec, corr_values) — corr_values trong [-1, 1].
    """
    window_n = max(2, int(round(window_sec * sr_grid)))
    hop_n = max(1, int(round(hop_sec * sr_grid)))
    n = len(sig_a)

    centers, corrs = [], []
    start = 0
    while start + window_n <= n:
        a = sig_a[start:start + window_n]
        b = sig_b[start:start + window_n]
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
        corr = float((a_c * b_c).sum() / denom) if denom > 1e-9 else 0.0
        center_idx = start + window_n // 2
        centers.append(center_idx / sr_grid)
        corrs.append(corr)
        start += hop_n

    if not centers:
        # video/audio quá ngắn hơn 1 cửa sổ: trả về tương quan toàn cục
        a_c = sig_a - sig_a.mean()
        b_c = sig_b - sig_b.mean()
        denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
        corr = float((a_c * b_c).sum() / denom) if denom > 1e-9 else 0.0
        centers = [n / (2 * sr_grid)]
        corrs = [corr]

    return np.array(centers), np.array(corrs)


def correlation_to_gain(
    corr_values: np.ndarray,
    low: float = 0.05,
    high: float = 0.35,
    min_gain: float = 0.08,
) -> np.ndarray:
    """
    Chuyển hệ số tương quan thành gain [min_gain, 1.0] bằng ánh xạ tuyến
    tính có ngưỡng. corr <= low -> min_gain; corr >= high -> 1.0.
    min_gain > 0 (không tắt hẳn về 0) để tránh mất tự nhiên/artefact âm
    thanh khi tương quan đo được thấp do nhiễu đo lường, không hẳn do im
    lặng thật.
    """
    g = (corr_values - low) / max(high - low, 1e-6)
    g = np.clip(g, 0.0, 1.0)
    g = min_gain + (1.0 - min_gain) * g
    return g


def smooth_gain(gain: np.ndarray, smooth_n: int = 3) -> np.ndarray:
    if smooth_n <= 1 or len(gain) < smooth_n:
        return gain
    kernel = np.ones(smooth_n) / smooth_n
    return np.convolve(gain, kernel, mode="same")


def apply_gain_envelope(
    samples: np.ndarray, sr: int, gain_times: np.ndarray, gain_values: np.ndarray,
    mode: str = "soft", hard_threshold: float = 0.5,
) -> np.ndarray:
    """
    Áp gain envelope (theo thời gian, giá trị 0..1) lên toàn bộ audio.
    mode='soft': nhân trực tiếp gain đã nội suy lên biên độ (giữ sắc thái).
    mode='hard': nhị phân hoá gain theo hard_threshold rồi làm mượt biên
                 (fade) để tránh tiếng click khi audio bật/tắt đột ngột.
    """
    n = len(samples)
    sample_times = np.arange(n) / sr
    gain_per_sample = np.interp(
        sample_times, gain_times, gain_values,
        left=gain_values[0] if len(gain_values) else 1.0,
        right=gain_values[-1] if len(gain_values) else 1.0,
    )

    if mode == "hard":
        binary = (gain_per_sample >= hard_threshold).astype(np.float64)
        fade_n = max(1, int(0.03 * sr))  # fade 30ms để tránh click
        kernel = np.ones(fade_n) / fade_n
        gain_per_sample = np.convolve(binary, kernel, mode="same")

    return (samples.astype(np.float64) * gain_per_sample).astype(np.float32)
