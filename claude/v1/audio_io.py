"""
audio_io.py
Trích xuất và đọc/ghi âm thanh từ file video bằng ffmpeg.
"""
from __future__ import annotations
import subprocess
import shutil
import numpy as np
import soundfile as sf


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Không tìm thấy ffmpeg trong PATH. Cài đặt: "
            "'sudo apt install ffmpeg' (Linux) hoặc 'brew install ffmpeg' (macOS), "
            "hoặc tải từ https://ffmpeg.org/download.html (Windows)."
        )


def extract_audio(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> str:
    """
    Trích xuất toàn bộ âm thanh từ video ra file WAV mono, PCM 16-bit.
    Trả về đường dẫn file wav đã tạo.
    """
    check_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                      # bỏ video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # sample rate
        "-acodec", "pcm_s16le",
        output_wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trích xuất âm thanh thất bại (mã {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )
    return output_wav_path


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Đọc file audio, trả về (samples float32 mono, sample_rate)."""
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def save_audio(path: str, samples: np.ndarray, sr: int) -> None:
    """Ghi mảng samples (float32, [-1,1]) ra file wav."""
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")
