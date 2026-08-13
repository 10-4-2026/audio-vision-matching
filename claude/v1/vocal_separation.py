"""
vocal_separation.py
Tách giọng nói (vocals) ra khỏi nhạc nền bằng Demucs (Meta AI).
Đây là bước TÙY CHỌN: nếu demucs/torch chưa được cài đặt, pipeline chính
(main.py) sẽ tự động bỏ qua bước này và cảnh báo cho người dùng, thay vì
báo lỗi toàn bộ.

Cài đặt (nặng, cần internet tốt, khuyến nghị GPU nhưng CPU vẫn chạy được
chỉ chậm hơn):
    pip install demucs
Lần chạy đầu tiên demucs sẽ tự tải model pretrained (~80MB) về.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys


class DemucsNotAvailable(Exception):
    pass


def is_demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def separate_vocals(input_wav: str, output_dir: str, model_name: str = "htdemucs") -> str:
    """
    Tách input_wav thành 2 stem: vocals.wav và no_vocals.wav (nhạc/tạp âm
    khác), lưu trong output_dir. Trả về đường dẫn tới file vocals.wav.

    Dùng subprocess gọi `python -m demucs` (CLI) thay vì import trực tiếp
    API nội bộ của demucs, để tránh phụ thuộc vào chi tiết API có thể đổi
    giữa các phiên bản.
    """
    if not is_demucs_available():
        raise DemucsNotAvailable(
            "Demucs chưa được cài đặt. Chạy 'pip install demucs' để bật bước "
            "tách nhạc nền (khuyến nghị), hoặc dùng --skip_vocal_separation "
            "để bỏ qua bước này (audio sẽ còn lẫn nhạc nền)."
        )

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model_name,
        "-o", output_dir,
        input_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs chạy thất bại (mã {result.returncode}):\n{result.stderr[-2000:]}"
        )

    base = os.path.splitext(os.path.basename(input_wav))[0]
    vocals_path = os.path.join(output_dir, model_name, base, "vocals.wav")
    if not os.path.isfile(vocals_path):
        raise RuntimeError(
            f"Demucs chạy xong nhưng không tìm thấy file kết quả tại: {vocals_path}\n"
            f"stdout: {result.stdout[-1000:]}"
        )
    return vocals_path
