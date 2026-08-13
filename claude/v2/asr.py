"""
asr.py
------
Nhận dạng giọng nói (Speech-to-Text) tiếng Việt trên các đoạn audio đã được
xác định là lời tài xế (từ ASD + tuỳ chọn diarization cross-check).

Dùng faster-whisper (bản tối ưu tốc độ của Whisper) với model large-v3, hỗ
trợ tốt tiếng Việt kèm code-switch tiếng Anh. Nếu cần độ chính xác tiếng Việt
cao hơn nữa, có thể thay bằng PhoWhisper (VinAI) qua thư viện `transformers`.
"""

from dataclasses import dataclass
import numpy as np
import soundfile as sf
import tempfile
import os
from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class VietnameseASR:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        # nếu không có GPU: device="cpu", compute_type="int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_clip(self, wav: np.ndarray, sr: int, start_offset: float = 0.0) -> list:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, sr)
            tmp_path = tmp.name
        try:
            segments, _info = self.model.transcribe(
                tmp_path,
                language="vi",              # ép ngôn ngữ chính là tiếng Việt
                task="transcribe",
                vad_filter=True,            # lọc khoảng lặng còn sót
                beam_size=5,
            )
            results = [
                TranscriptSegment(
                    start=start_offset + seg.start,
                    end=start_offset + seg.end,
                    text=seg.text.strip(),
                )
                for seg in segments
            ]
        finally:
            os.remove(tmp_path)
        return results

    def transcribe_segments(self, clips: list) -> list:
        """clips: list[(start_sec, end_sec, wav_ndarray)] từ audio_utils.cut_segments"""
        all_segments = []
        for start, _end, wav in clips:
            if len(wav) < 400:  # quá ngắn, bỏ qua nhiễu
                continue
            all_segments.extend(self.transcribe_clip(wav, sr=16000, start_offset=start))
        return all_segments


def format_transcript(segments: list) -> str:
    lines = []
    for seg in segments:
        ts = f"[{seg.start:6.2f}s - {seg.end:6.2f}s]"
        lines.append(f"{ts} {seg.text}")
    return "\n".join(lines)
