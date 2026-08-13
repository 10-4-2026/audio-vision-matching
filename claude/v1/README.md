# Trích xuất lời nói tài xế từ video (khẩu hình + audio hỗn tạp)

Pipeline cho bài toán: video quay tài xế (camera trong xe VinFast, chỉ có
1 người — tài xế — trong khung hình, tài xế **không nhất thiết** nhìn
thẳng camera), âm thanh trong xe lẫn nhạc, tiếng hành khách, tạp âm...
Mục tiêu: lấy ra nội dung lời nói **của riêng tài xế** dưới dạng văn bản
tiếng Việt (có thể xen từ tiếng Anh).

## Kiến trúc pipeline

```
video.mp4
   │
   ├─► [1] audio_io.py        : trích audio ra wav (ffmpeg)
   │
   ├─► [2] vocal_separation.py: (tùy chọn) Demucs tách bỏ nhạc nền → vocals.wav
   │
   ├─► [3] mouth_detector.py  : mediapipe FaceLandmarker → tín hiệu chuyển
   │                            động khẩu hình tài xế theo thời gian (MAR)
   │
   ├─► [4] av_gate.py         : tương quan MAR ↔ năng lượng audio theo cửa
   │                            sổ trượt → gain mask → cô lập giọng tài xế
   │                            khỏi giọng hành khách còn sót lại
   │
   └─► [5] transcribe.py      : faster-whisper → transcript.txt / .json
```

### Vì sao dùng "tương quan khẩu hình – âm thanh" thay vì tách giọng bằng AI riêng?

Các model audio-visual speech separation "xịn" (VisualVoice, Looking-to-
Listen...) cho chất lượng tốt hơn nhưng đòi hỏi GPU mạnh + model pretrained
lớn, khó triển khai nhanh. Cách tiếp cận ở đây nhẹ hơn nhiều: khi tài xế
đang nói, chuyển động môi và năng lượng âm thanh biến thiên đồng bộ; khi
âm thanh đến từ hành khách/nhạc, hai tín hiệu này không tương quan. Tính
hệ số tương quan cục bộ theo cửa sổ trượt, rồi dùng nó làm "cổng" (gate)
để giữ/giảm âm lượng theo thời gian — không cần huấn luyện hay tải model
tách giọng nặng.

**Hạn chế cần biết:** đây là heuristic, không phải deep-learning
separation thật sự. Nếu hành khách nói *đúng lúc* tài xế cũng đang mấp máy
môi (ví dụ đang nhai, ngáp), tín hiệu có thể sai. Nếu hành khách nói với
âm lượng lớn hơn nhiều so với tài xế, phần dư âm có thể vẫn lọt qua ở mức
`min_gain`. Có thể tinh chỉnh các tham số `--corr_low/--corr_high/--min_gain`
để cân bằng giữa "giữ được giọng tài xế" và "chặn được giọng khác".

## Cài đặt

```bash
pip install -r requirements.txt
```

**ffmpeg** phải có sẵn trong hệ thống (không cài qua pip):
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: tải tại https://ffmpeg.org/download.html

**Demucs (tùy chọn nhưng khuyến nghị mạnh)** — tách nhạc nền, giúp bước 4
chính xác hơn nhiều vì chỉ cần phân biệt tài xế vs hành khách, không phải
lo cả nhạc:
```bash
pip install demucs
```
Nặng (~2-3GB do kéo theo torch), lần đầu chạy sẽ tự tải model pretrained
(~80MB). Nếu không cài, pipeline vẫn chạy được (dùng `--skip_vocal_separation`
hoặc để mặc định — code tự phát hiện thiếu demucs và cảnh báo + bỏ qua),
chỉ là audio đầu ra sẽ còn lẫn nhạc.

**Model khuôn mặt** (`face_landmarker.task`, ~3.7MB) được tự động tải về
thư mục `models/` ở lần chạy đầu. Nếu mạng của bạn chặn
`storage.googleapis.com`, tải thủ công tại:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
rồi đặt vào `models/face_landmarker.task`, hoặc dùng `--model_path`.

**faster-whisper** sẽ tự tải model ASR (từ Hugging Face) ở lần chạy đầu —
cần internet. Model `large-v3` cho chất lượng tốt nhất nhưng nặng và chậm
trên CPU; nếu máy không có GPU, cân nhắc dùng `--whisper_model medium`
hoặc `small` để chạy nhanh hơn.

## Cách dùng

Chạy đầy đủ (khuyến nghị, có tách nhạc):
```bash
python main.py --input driver_video.mp4 --workdir ./work
```

Bỏ qua tách nhạc (chạy nhanh hơn, không cần demucs):
```bash
python main.py --input driver_video.mp4 --workdir ./work --skip_vocal_separation
```

Chạy nhanh trên máy yếu (model ASR nhỏ, xử lý bớt frame video):
```bash
python main.py --input driver_video.mp4 --workdir ./work \
  --whisper_model small --frame_stride 2
```

Xuất thêm file debug (tín hiệu khẩu hình, gain envelope) để kiểm tra/tinh
chỉnh:
```bash
python main.py --input driver_video.mp4 --workdir ./work --save_debug_csv
```

## Output (trong thư mục `--workdir`)

| File | Nội dung |
|---|---|
| `audio_raw.wav` | Audio gốc trích từ video |
| `demucs_out/.../vocals.wav` | (nếu bật tách nhạc) audio chỉ còn giọng người |
| `audio_driver_isolated.wav` | Audio sau khi cô lập giọng tài xế — **nghe thử file này để đánh giá chất lượng tách** |
| `transcript.txt` | Văn bản lời nói của tài xế |
| `transcript.json` | Văn bản kèm timestamp từng đoạn (`start`, `end`, `text`) |
| `mouth_signal.csv` | (nếu `--save_debug_csv`) tín hiệu MAR theo thời gian |
| `gain_envelope.csv` | (nếu `--save_debug_csv`) hệ số tương quan + gain theo thời gian |

## Tham số quan trọng cần tinh chỉnh theo dữ liệu thực tế

- `--corr_low` / `--corr_high`: ngưỡng tương quan map sang gain 0→1. Nếu
  giọng tài xế vẫn bị cắt nhầm (mất từ), thử **giảm** `--corr_low`. Nếu
  giọng hành khách vẫn lọt qua nhiều, thử **tăng** `--corr_high`.
- `--min_gain`: gain sàn khi tương quan thấp. Đặt 0 để tắt hẳn các đoạn
  không liên quan (dứt khoát hơn nhưng dễ nghe "cụt" nếu tương quan đo sai);
  giữ > 0 (mặc định 0.08) cho tự nhiên hơn.
- `--gate_mode soft|hard`: `soft` giữ sắc thái âm lượng liên tục (khuyến
  nghị mặc định); `hard` bật/tắt dứt khoát theo `--hard_threshold`, phù hợp
  khi cần cắt triệt để giọng hành khách với giá phải trả là âm thanh kém
  tự nhiên hơn.
- `--frame_stride`: video dashcam thường không cần xử lý mọi frame để bắt
  chuyển động môi (vốn chậm hơn nhiều so với fps video); tăng lên 2-3 để
  chạy nhanh hơn đáng kể mà ít ảnh hưởng chất lượng.

## Gợi ý cải thiện thêm (nếu cần độ chính xác cao hơn)

1. **Diarization + đối chiếu**: dùng `pyannote.audio` để tách các đoạn
   theo từng giọng nói (không cần biết là ai), rồi chọn đoạn nào tương
   quan tốt nhất với MAR là giọng tài xế — chính xác hơn cách gating liên
   tục hiện tại nhưng cần thêm setup (tài khoản Hugging Face + token).
2. **Model audio-visual separation chuyên dụng** (VisualVoice, AV-TSE...)
   nếu có GPU mạnh và chấp nhận độ phức tạp triển khai cao hơn.
3. Nếu camera lắp cố định đúng vị trí tài xế (không đổi góc), có thể crop
   sẵn vùng ROI khuôn mặt để tăng tốc + độ ổn định phát hiện.
