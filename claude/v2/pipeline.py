"""
pipeline.py
-----------
Điều phối toàn bộ quy trình:

  video -> [1] track khuôn mặt (uniface) -> chọn track tài xế
        -> [2] ASD (khẩu hình + audio) -> các đoạn thời gian tài xế đang nói
        -> [3] (tuỳ chọn) diarization cross-check để loại nhiễu chồng tiếng
        -> [4] cắt audio theo các đoạn đó
        -> [5] ASR tiếng Việt (faster-whisper) -> văn bản lời tài xế

Chạy:
    python pipeline.py --video path/to/video.mp4 --out transcript.txt \
        --asd_checkpoint path/to/light_asd.pth --driver_side left
"""

import argparse
import cv2
import numpy as np

from face_track import DriverFaceTracker, select_driver_track, crop_mouth
from asd import LightASDWrapper, run_asd, scores_to_segments
from audio_utils import extract_audio, diarize_and_crosscheck, cut_segments
from asr import VietnameseASR, format_transcript


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def build_mouth_crop_sequence(video_path: str, track, fps: float) -> np.ndarray:
    """
    Dựng chuỗi ảnh crop miệng liên tục theo từng frame video (kể cả frame
    track bị mất detection thì lặp lại crop gần nhất, để giữ độ dài chuỗi
    khớp với audio feature).
    """
    det_by_frame = {d.frame_idx: d for d in track.detections}
    max_frame = max(det_by_frame.keys())

    cap = cv2.VideoCapture(video_path)
    crops = []
    last_det = None
    frame_idx = 0
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        det = det_by_frame.get(frame_idx, last_det)
        if det is not None:
            crops.append(crop_mouth(frame, det))
            last_det = det
        else:
            crops.append(np.zeros((112, 112, 3), dtype=np.uint8))
        frame_idx += 1
    cap.release()
    return np.stack(crops, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Trích văn bản lời nói của tài xế từ video.")
    parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--out", default="transcript.txt", help="File văn bản kết quả")
    parser.add_argument("--asd_checkpoint", required=True, help="Checkpoint model Light-ASD (.pth)")
    parser.add_argument("--driver_side", default="left", choices=["left", "right"],
                         help="Vị trí tài xế trong khung hình (tuỳ camera lắp đặt)")
    parser.add_argument("--asd_threshold", type=float, default=0.5)
    parser.add_argument("--use_diarization", action="store_true",
                         help="Bật đối chiếu diarization để loại nhiễu chồng tiếng")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token cho pyannote (nếu dùng)")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fps = get_video_fps(args.video)
    print(f"[1/5] Theo dõi khuôn mặt trong video (fps={fps:.2f}) ...")
    tracker = DriverFaceTracker()
    tracks = tracker.run(args.video)

    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    driver_track = select_driver_track(tracks, frame_width, driver_side=args.driver_side)
    print(f"  -> Chọn track tài xế: track_id={driver_track.track_id}, "
          f"{len(driver_track.detections)} detection")

    print("[2/5] Trích audio từ video ...")
    wav, sr = extract_audio(args.video, "_tmp_audio.wav")

    print("[3/5] Dựng chuỗi crop miệng & chạy Active Speaker Detection ...")
    mouth_crops = build_mouth_crop_sequence(args.video, driver_track, fps)
    asd_model = LightASDWrapper(checkpoint_path=args.asd_checkpoint, device=args.device)
    scores = run_asd(mouth_crops, wav, sr, fps, asd_model)
    segments = scores_to_segments(scores, fps, threshold=args.asd_threshold)
    print(f"  -> Tìm được {len(segments)} đoạn tài xế đang nói")

    if args.use_diarization:
        print("[3.5/5] Đối chiếu diarization để loại nhiễu chồng tiếng ...")
        segments = diarize_and_crosscheck("_tmp_audio.wav", segments, hf_token=args.hf_token)
        print(f"  -> Còn lại {len(segments)} đoạn sau khi lọc")

    print("[4/5] Cắt audio theo các đoạn đã xác định ...")
    clips = cut_segments(wav, sr, segments)

    print(f"[5/5] Nhận dạng giọng nói tiếng Việt (model={args.whisper_model}) ...")
    asr = VietnameseASR(model_size=args.whisper_model, device=args.device)
    transcript_segments = asr.transcribe_segments(clips)

    transcript_text = format_transcript(transcript_segments)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print(f"\nHoàn tất. Kết quả lưu tại: {args.out}\n")
    print(transcript_text)


if __name__ == "__main__":
    main()
