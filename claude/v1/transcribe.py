"""
transcribe.py
Chuyển âm thanh (đã làm sạch) thành văn bản bằng faster-whisper.
Hỗ trợ tiếng Việt xen tiếng Anh (code-switching) — Whisper multilingual
xử lý khá tốt việc này, có thể hỗ trợ thêm bằng initial_prompt.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json


@dataclass
class Segment:
    start: float
    end: float
    text: str


# Gợi ý các từ tiếng Anh người Việt hay dùng xen trong câu nói, giúp
# Whisper thiên về nhận đúng các từ này thay vì "Việt hoá" âm gần giống.
DEFAULT_INITIAL_PROMPT = (
    "Đoạn hội thoại tiếng Việt, thỉnh thoảng có xen từ tiếng Anh như: "
    "ok, oke, deadline, meeting, email, laptop, wifi, app, video call, "
    "facebook, note, plan, feedback, report, team, boss, sếp."
)


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
    language: str = "vi",
    vad_filter: bool = True,
    initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
    beam_size: int = 5,
) -> dict:
    """
    Trả về:
    {
        'text': str (toàn bộ transcript nối lại),
        'segments': list[Segment] (có timestamp),
        'language': str,
        'language_probability': float,
    }
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    seg_iter, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=initial_prompt,
        beam_size=beam_size,
    )

    segments = []
    full_text_parts = []
    for seg in seg_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=round(seg.start, 2), end=round(seg.end, 2), text=text))
        full_text_parts.append(text)

    return {
        "text": " ".join(full_text_parts).strip(),
        "segments": segments,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
    }


def save_transcript(result: dict, txt_path: str, json_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"] + "\n")

    payload = {
        "text": result["text"],
        "language": result["language"],
        "language_probability": result["language_probability"],
        "segments": [asdict(s) for s in result["segments"]],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
