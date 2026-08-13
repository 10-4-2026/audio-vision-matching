# Pipeline: Trích văn bản lời nói của tài xế từ video cabin

## Ý tưởng cốt lõi
Không dùng khẩu hình để "đọc ra chữ" trực tiếp (lipreading → text gần như
không khả thi cho tiếng Việt, thiếu dữ liệu training). Thay vào đó, khẩu hình
được dùng cho **Active Speaker Detection (ASD)**: xác định khoảng thời gian
nào môi tài xế khớp với âm thanh đang phát ra, để **tách đúng đoạn audio của
tài xế** ra khỏi những người khác nói cùng lúc. Sau đó chạy ASR (Whisper)
trên riêng các đoạn đó.

```
video ──► [face_track.py]  detect + track khuôn mặt (uniface) ──► chọn track tài xế
                                                                       │
video + audio ──► [asd.py]  Active Speaker Detection (Light-ASD) ──► timeline "đang nói / không"
                                                                       │
                                                          [audio_utils.py] cắt audio theo timeline
                                                          (tuỳ chọn: đối chiếu pyannote diarization
                                                           để loại nhiễu khi nhiều người nói chồng)
                                                                       │
                                                          [asr.py] faster-whisper (language="vi")
                                                                       │
                                                              văn bản lời tài xế
```

## Cài đặt
```bash
pip install -r requirements.txt

# ffmpeg (trích audio)
sudo apt-get install ffmpeg

# Model ASD - chọn 1 trong 2, clone về và tải checkpoint pretrained:
git clone https://github.com/Junhua-Liao/Light-ASD      # nhẹ, nhanh (khuyến nghị)
# hoặc
git clone https://github.com/TaoRuijie/TalkNet-ASD       # chính xác hơn, chậm hơn
```

Sau khi clone, mở `asd.py`, sửa phần `TODO` trong `LightASDWrapper.__init__`
để import đúng class model của repo và load checkpoint `.pth` bạn đã tải.

## Chạy
```bash
python pipeline.py \
    --video duong_dan_video.mp4 \
    --asd_checkpoint Light-ASD/weight/pretrain_AVA_CVPR.pth \
    --driver_side left \
    --out transcript.txt \
    --device cuda \
    --use_diarization --hf_token <HUGGINGFACE_TOKEN>
```

Tham số quan trọng:
- `--driver_side`: vị trí tài xế trong khung hình (`left`/`right`). Cần chỉnh
  theo cách lắp camera thực tế của bạn. Nếu camera chỉ quay đúng 1 người
  (tài xế), có thể bỏ qua bước lọc vị trí và lấy track dài nhất.
- `--use_diarization`: bật nếu video có nhiều người nói **chồng tiếng** cùng
  lúc, giúp loại các trường hợp ASD nhận nhầm do môi mấp máy nhưng tiếng
  thực chất từ người khác vọng vào.
- `--whisper_model`: `large-v3` cho độ chính xác cao nhất; dùng `medium`
  hoặc `small` nếu cần tốc độ / chạy CPU.

## Các điểm cần lưu ý khi triển khai thực tế
1. **Vị trí camera quyết định logic chọn track tài xế** (`select_driver_track`
   trong `face_track.py`) — hãy chỉnh heuristic này theo góc quay thực tế,
   hoặc đơn giản hơn: cho người dùng click chọn khuôn mặt tài xế ở frame đầu.
2. **Chất lượng ASD phụ thuộc độ phân giải & góc quay khuôn mặt** — nếu
   khuôn mặt tài xế quá nhỏ/bị che (khẩu trang, góc nghiêng), độ chính xác
   ASD giảm; nên có bước fallback dùng speaker diarization đơn thuần.
3. **Whisper xử lý code-switch tiếng Việt-Anh khá tốt** nhưng vẫn có thể lẫn
   với phương ngữ vùng miền hoặc từ lóng; cân nhắc fine-tune thêm nếu cần độ
   chính xác cao cho domain giao thông/vận tải.
4. Toàn bộ code trong repo này là khung sườn (scaffold) đầy đủ logic điều
   phối — phần **load checkpoint model ASD** cần bạn tự hoàn thiện theo đúng
   API của repo Light-ASD/TalkNet-ASD vì weight file không nằm trong phạm vi
   tải tự động ở đây.
