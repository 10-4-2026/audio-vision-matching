"""
main.py
Pipeline hoàn chỉnh: video tài xế (1 người, không cần nhìn camera, âm
thanh hỗn tạp) -> văn bản lời nói của tài xế.

Các bước:
  1. Trích xuất audio từ video (ffmpeg)
  2. [Tùy chọn, khuyến nghị] Tách nhạc nền bằng Demucs -> giữ lại vocals
  3. Trích xuất tín hiệu chuyển động khẩu hình (MAR) của tài xế theo thời gian
  4. Tương quan MAR với năng lượng audio -> gain mask -> cô lập giọng tài xế
     khỏi giọng hành khách / tạp âm còn sót lại
  5. ASR (faster-whisper) -> văn bản tiếng Việt (có xen tiếng Anh)

Cách dùng cơ bản:
    python main.py --input driver_video.mp4 --workdir ./work

Bỏ qua bước tách nhạc (nếu chưa cài demucs hoặc muốn chạy nhanh):
    python main.py --input driver_video.mp4 --workdir ./work --skip_vocal_separation
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import csv

from audio_io import extract_audio, load_audio, save_audio
from mouth_detector import extract_mouth_signal
from av_gate import (
    compute_audio_envelope, resample_to_grid, zscore,
    windowed_correlation, correlation_to_gain, smooth_gain, apply_gain_envelope,
)
from vocal_separation import separate_vocals, is_demucs_available, DemucsNotAvailable
from transcribe import transcribe, save_transcript, DEFAULT_INITIAL_PROMPT


def log(msg: str) -> None:
    print(f"[main] {msg}", file=sys.stderr)


def run(args: argparse.Namespace) -> dict:
    os.makedirs(args.workdir, exist_ok=True)
    t0 = time.time()

    # ---- 1. Trích xuất audio ----
    log("Bước 1/5: Trích xuất audio từ video...")
    raw_wav = os.path.join(args.workdir, "audio_raw.wav")
    extract_audio(args.input, raw_wav, sample_rate=16000)

    # ---- 2. Tách nhạc nền (tùy chọn) ----
    vocals_wav = raw_wav
    if not args.skip_vocal_separation:
        log("Bước 2/5: Tách nhạc nền bằng Demucs (giữ lại vocals)...")
        if is_demucs_available():
            try:
                vocals_wav = separate_vocals(raw_wav, os.path.join(args.workdir, "demucs_out"))
                log(f"  -> Đã tách xong: {vocals_wav}")
            except Exception as e:
                log(f"  CẢNH BÁO: Tách nhạc thất bại ({e}). Dùng audio gốc thay thế.")
                vocals_wav = raw_wav
        else:
            log("  CẢNH BÁO: Demucs chưa được cài (pip install demucs). "
                "Bỏ qua bước tách nhạc, audio còn lẫn nhạc nền.")
    else:
        log("Bước 2/5: Bỏ qua tách nhạc nền (--skip_vocal_separation).")

    samples, sr = load_audio(vocals_wav)

    # ---- 3. Tín hiệu khẩu hình tài xế ----
    log("Bước 3/5: Trích xuất tín hiệu khẩu hình (MAR) từ video...")
    mouth = extract_mouth_signal(
        args.input, sample_stride=args.frame_stride, model_path=args.model_path,
    )
    log(f"  -> Tỉ lệ frame phát hiện được mặt tài xế: {mouth['detected_ratio']:.1%}")
    if mouth["detected_ratio"] < 0.3:
        log("  CẢNH BÁO: Tỉ lệ phát hiện mặt thấp — kiểm tra lại góc đặt camera "
            "hoặc ánh sáng, kết quả tách giọng có thể kém chính xác.")

    if args.save_debug_csv:
        _save_csv(
            os.path.join(args.workdir, "mouth_signal.csv"),
            ["time_sec", "mar_raw", "mar_smooth"],
            zip(mouth["times"], mouth["mar_raw"], mouth["mar_smooth"]),
        )

    # ---- 4. Tương quan audio-visual -> gain mask -> áp lên audio ----
    log("Bước 4/5: Tính tương quan khẩu hình-âm thanh và cô lập giọng tài xế...")
    env_times, env = compute_audio_envelope(samples, sr)
    mar_on_env_grid = resample_to_grid(mouth["times"], mouth["mar_smooth"], env_times)

    a = zscore(mar_on_env_grid)
    b = zscore(env)
    sr_grid = 1.0 / max(1e-6, float((env_times[-1] - env_times[0]) / max(1, len(env_times) - 1)))

    corr_times, corr_values = windowed_correlation(
        a, b, sr_grid, window_sec=args.corr_window_sec, hop_sec=args.corr_hop_sec,
    )
    gain = correlation_to_gain(
        corr_values, low=args.corr_low, high=args.corr_high, min_gain=args.min_gain,
    )
    gain = smooth_gain(gain, smooth_n=args.gain_smooth_n)

    gated_samples = apply_gain_envelope(
        samples, sr, corr_times, gain, mode=args.gate_mode, hard_threshold=args.hard_threshold,
    )

    gated_wav = os.path.join(args.workdir, "audio_driver_isolated.wav")
    save_audio(gated_wav, gated_samples, sr)
    log(f"  -> Đã ghi audio đã cô lập giọng tài xế: {gated_wav}")

    if args.save_debug_csv:
        _save_csv(
            os.path.join(args.workdir, "gain_envelope.csv"),
            ["time_sec", "correlation", "gain"],
            zip(corr_times, corr_values, gain),
        )

    # ---- 5. ASR ----
    log(f"Bước 5/5: Nhận dạng giọng nói (faster-whisper, model={args.whisper_model})...")
    result = transcribe(
        gated_wav,
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
        language=args.language,
        initial_prompt=None if args.no_prompt else DEFAULT_INITIAL_PROMPT,
    )

    txt_path = os.path.join(args.workdir, "transcript.txt")
    json_path = os.path.join(args.workdir, "transcript.json")
    save_transcript(result, txt_path, json_path)

    elapsed = time.time() - t0
    log(f"Hoàn tất trong {elapsed:.1f}s.")
    log(f"Transcript (txt): {txt_path}")
    log(f"Transcript (json, có timestamp): {json_path}")
    print("\n=== NỘI DUNG LỜI NÓI CỦA TÀI XẾ ===")
    print(result["text"] if result["text"] else "(không nhận dạng được nội dung nào)")

    return {
        "transcript": result,
        "gated_audio_path": gated_wav,
        "mouth_detected_ratio": mouth["detected_ratio"],
    }


def _save_csv(path: str, header: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trích xuất lời nói của tài xế từ video (khẩu hình + audio hỗn tạp)."
    )
    p.add_argument("--input", required=True, help="Video đầu vào (chỉ có 1 người: tài xế)")
    p.add_argument("--workdir", default="./work", help="Thư mục chứa file trung gian + kết quả")

    # Vocal separation
    p.add_argument("--skip_vocal_separation", action="store_true",
                    help="Bỏ qua bước tách nhạc nền bằng Demucs")

    # Mouth detection
    p.add_argument("--frame_stride", type=int, default=1,
                    help="Chỉ xử lý 1/N frame video để tăng tốc (mặc định 1 = mọi frame)")
    p.add_argument("--model_path", default=None,
                    help="Đường dẫn model face_landmarker.task đã tải sẵn (tùy chọn)")

    # AV correlation gating
    p.add_argument("--corr_window_sec", type=float, default=1.0,
                    help="Độ dài cửa sổ trượt tính tương quan (giây)")
    p.add_argument("--corr_hop_sec", type=float, default=0.25,
                    help="Bước nhảy cửa sổ trượt (giây)")
    p.add_argument("--corr_low", type=float, default=0.05,
                    help="Ngưỡng tương quan dưới -> gain tối thiểu")
    p.add_argument("--corr_high", type=float, default=0.35,
                    help="Ngưỡng tương quan trên -> gain tối đa (1.0)")
    p.add_argument("--min_gain", type=float, default=0.08,
                    help="Gain tối thiểu khi tương quan thấp (0 = tắt hẳn, >0 tránh artefact)")
    p.add_argument("--gain_smooth_n", type=int, default=3,
                    help="Số điểm làm mượt gain envelope")
    p.add_argument("--gate_mode", choices=["soft", "hard"], default="soft",
                    help="soft = nhân gain liên tục; hard = bật/tắt nhị phân có fade")
    p.add_argument("--hard_threshold", type=float, default=0.5,
                    help="Ngưỡng gain để bật/tắt khi --gate_mode hard")

    # ASR
    p.add_argument("--whisper_model", default="large-v3",
                    help="Kích thước model faster-whisper (tiny/base/small/medium/large-v3...)")
    p.add_argument("--whisper_device", default="auto", help="cpu/cuda/auto")
    p.add_argument("--whisper_compute_type", default="default",
                    help="vd: int8, float16, default")
    p.add_argument("--language", default="vi", help="Mã ngôn ngữ chính (mặc định 'vi')")
    p.add_argument("--no_prompt", action="store_true",
                    help="Không dùng initial_prompt gợi ý từ tiếng Anh thường gặp")

    p.add_argument("--save_debug_csv", action="store_true",
                    help="Lưu thêm CSV debug (mouth_signal.csv, gain_envelope.csv)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
