"""
audio_utils.py
--------------
Tiện ích trích xuất audio từ video, và (tùy chọn) đối chiếu với kết quả
speaker diarization để xử lý trường hợp nhiều người nói CHỒNG TIẾNG nhau
(overlap) - giúp tăng độ tin cậy khi xác định đoạn nào thực sự là tài xế nói.
"""

import subprocess
import numpy as np
import soundfile as sf
import librosa


def extract_audio(video_path: str, out_wav_path: str, sr: int = 16000):
    """Dùng ffmpeg trích audio mono 16kHz từ video (khuyến nghị cho ASR)."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr), "-vn", out_wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    wav, sr_out = sf.read(out_wav_path)
    return wav.astype(np.float32), sr_out


def load_audio(path: str, sr: int = 16000):
    wav, orig_sr = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def diarize_and_crosscheck(wav_path: str, driver_segments: list, hf_token: str = None):
    """
    (Tùy chọn) Dùng pyannote.audio để tách các người nói khác nhau trong audio,
    sau đó chỉ giữ lại phần trong driver_segments (từ ASD) mà trùng khớp với
    MỘT speaker nhất quán (loại bỏ trường hợp môi tài xế "mấp máy" nhưng tiếng
    thực chất là của người khác chồng vào, ví dụ cười nói ở ghế sau).

    Cần: pip install pyannote.audio, và token HuggingFace nếu model yêu cầu.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization = pipeline(wav_path)

    # gộp các đoạn diarization theo speaker
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    # với mỗi đoạn ASD của tài xế, tìm speaker diarization overlap nhiều nhất
    refined = []
    for seg in driver_segments:
        best_speaker, best_overlap = None, 0.0
        for spk, segs in speaker_segments.items():
            total = sum(overlap(seg, s) for s in segs)
            if total > best_overlap:
                best_speaker, best_overlap = spk, total
        seg_len = seg[1] - seg[0]
        # chỉ giữ nếu >50% thời lượng đoạn khớp với đúng 1 speaker
        if best_overlap / max(seg_len, 1e-6) > 0.5:
            refined.append(seg)
    return refined


def cut_segments(wav: np.ndarray, sr: int, segments: list):
    """Cắt waveform theo danh sách (start_sec, end_sec)."""
    clips = []
    for start, end in segments:
        s_idx, e_idx = int(start * sr), int(end * sr)
        clips.append((start, end, wav[s_idx:e_idx]))
    return clips
