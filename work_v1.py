import cv2
import mediapipe as mp

import numpy as np

import subprocess
import av
import sys
import urllib.request

from pathlib import Path

from faster_whisper import WhisperModel



# Thiết lập encoding UTF-8 cho console trên Windows

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if hasattr(sys.stderr,'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')







# ==========================================

# 1. BÓC TÁCH KHẨU HÌNH BẰNG MEDIAPIPE

# ==========================================

class LipROIExtractor:

    """Class hỗ trợ phát hiện khuôn mặt và crop riêng vùng khẩu hình (Lips)."""    

    # Danh sách các chỉ số Landmark tương ứng với môi trong MediaPipe Face Mesh
    LIP_LANDMARKS  = [
        61,146,91,181,84,17,314,405,321,375,291,
        308,324,318,402,317,14,87,178,88,95
        ]

    def __init__(self, target_size=(96, 96), padding_factor=0.3):
        self.target_size = target_size
        self.padding_factor = padding_factor
        # Tải mô hình Face Landmarker nếu chưa có
        model_path = Path("face_landmarker.task")
        if not model_path.exists():
            print("Đang tải model face_landmarker.task từ Google...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, model_path)
            print("Đã tải xong model!")            

        from mediapipe.tasks  import  python
        from mediapipe.tasks.python import vision

        base_options =  python.BaseOptions(model_asset_path=str(model_path))
        options =  vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
            )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_lip_roi(self, frame):
        """Trích xuất và resize vùng môi từ 1 frame ảnh."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results  = self.detector.detect(mp_image)
        if not results.face_landmarks:
            # Nếu không tìm thấy mặt (tài xế bị che mặt/ngoảnh đầu), trả về khung ảnh trống
            return np.zeros((self.target_size[1],self.target_size[0],3),dtype=np.uint8)

        # Lấy các điểm tọa độ môi
        face_landmarks = results.face_landmarks[0]
        lip_pts = []
        for idx in self.LIP_LANDMARKS:
            lm = face_landmarks[idx]
            lip_pts.append((int(lm.x * w), int(lm.y* h)))

        lip_pts = np.array(lip_pts)
        # Tính toán Bounding Box bao quanh vùng môi

        x_min, y_min = np.min(lip_pts,axis=0)
        x_max, y_max = np.max(lip_pts,axis=0)

        # Thêm padding xung quanh vùng môi

        box_w =  x_max - x_min
        box_h = y_max - y_min
        pad_x = int(box_w * self.padding_factor)
        pad_y = int(box_h * self.padding_factor)
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        # Crop và Resize về kích thước cố định (ví dụ 96x96)

        lip_crop = frame[y1:y2, x1:x2]
        if lip_crop.size == 0:
            return np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

        lip_resized = cv2.resize(lip_crop, self.target_size, interpolation=cv2.INTER_CUBIC)
        return lip_resized

# ==========================================

# 2. XỬ LÝ ÂM THANH TỪ VIDEO

# ==========================================

def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"): 
    """Dùng PyAV để bóc tách file âm thanh 16kHz Mono từ Video."""
    input_container =  av.open(video_path) 
    audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
    if not audio_stream:
        raise ValueError("Không tìm thấy luồng âm thanh trong video.")

    output_container =  av.open(output_audio_path, 'w')    

    # Thiết lập output stream định dạng pcm_s16le wav

    out_stream =  output_container.add_stream('pcm_s16le', rate=16000)
    out_stream.layout =  'mono'

    resampler =  av.AudioResampler(format='s16', layout='mono', rate=16000, )
      
    for packet in input_container.demux(audio_stream):
        for frame in packet.decode():
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                if not isinstance(resampled_frames, list) and not isinstance(resampled_frames, tuple):
                    resampled_frames = [resampled_frames]
                for rf in resampled_frames:
                    for out_packet in out_stream.encode(rf):
                        output_container.mux(out_packet)                        

    # Flush bộ mã hóa
    for out_packet in out_stream.encode():
        output_container.mux(out_packet)        

    output_container.close()
    input_container.close()
    return output_audio_path



# ==========================================

# 3. MÔ HÌNH LỌC TÁCH ÂM THANH MỤC TIÊU (AV-TSE)

# ==========================================

def apply_av_target_speaker_extraction(lip_frames, raw_audio_path):
    """
    Giả lập/Gọi mô hình AV-TSE (như VisualVoice / AV-Conv-TasNet / CTCNet)
    - Đầu vào: Chuỗi frame môi [T, 96, 96, 3] + Audio thô hỗn hợp
    - Đầu ra: File audio sạch đã lọc riêng giọng tài xế dựa theo nhịp cử động môi.
    """

    print(f"-> Đang xử lý lọc giọng bằng {len(lip_frames)}  khung hình khẩu hình...")    

    # [NOTE]: Trong triển khai thực tế, bạn sẽ pass `lip_frames` và `raw_audio_path`

    # vào mô hình AV-TSE pre-trained tại đây.

    clean_audio_path =  "driver_clean_voice.wav"    

    # Tạm thời trả về file audio gốc nếu chưa gắn trọng số model AV-TSE

    return raw_audio_path



# ==========================================

# 4. PIPELINE CHÍNH (MAIN WORKFLOW)

# ==========================================

def process_driver_video(video_path):
    print("=== BẮT ĐẦU QUY TRÌNH XỬ LÝ VIDEO TÀI XẾ ===")   
    # Bước 1: Trích xuất âm thanh từ Video

    print("[1/4] Đang tách Audio thô từ Video...")
    raw_audio_path =  extract_audio_from_video(video_path)

    # Bước 2: Đọc Video & Crop vùng khẩu hình bằng MediaPipe
    print("[2/4] Đang trích xuất chuỗi khẩu hình (Lip ROI)...")

    cap =  cv2.VideoCapture(video_path)
    extractor =  LipROIExtractor(target_size=(96, 96))

    lip_frames = [] 
    frame_count = 0

    while cap.isOpened():
        ret, frame =  cap.read()
        if not ret:
            break           

        lip_roi = extractor.extract_lip_roi(frame)
        lip_frames.append(lip_roi)
        frame_count += 1
    cap.release()

    print(f"-> Đã xử lý {frame_count} frames video.")
    

    # Bước 3: Đưa Audio + Lip Frames vào Mô hình AV-TSE để tách giọng sạch

    print("[3/4] Đang lọc bỏ tạp âm và giọng nói đè bằng AV-TSE...")

    clean_audio =  apply_av_target_speaker_extraction(lip_frames,   raw_audio_path)

    # Bước 4: Chuyển âm thanh sạch thành văn bản bằng Faster-Whisper

    print("[4/4] Đang chuyển giọng nói sạch thành văn bản (Speech-to-Text)...")

    device  =  "cuda" if cv2.cuda.getCudaEnabledDeviceCount()  >  0  else  "cpu" 
    compute_type =  "float16" if device == "cuda" else "float32"

    #whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)
    model_path = "./mylocalmodels/"
    whisper_model = WhisperModel(model_path, device=device, compute_type=compute_type)
    #segments, info = whisper_model.transcribe(clean_audio, language="vi")    
    segments, info = whisper_model.transcribe(clean_audio, language="en")    

    print("\n" +  "="*40) 
    print("KẾT QUẢ NHẬN DẠNG NỘI DUNG TÀI XẾ NÓI:")
    print("="*40)   

    full_transcript  =  "" 
    for segment in segments:
        text = f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}"
        print(text)
        full_transcript += segment.text + " "
    return full_transcript





# ==========================================

# CHẠY THỬ NGHIỆM

# ==========================================

if __name__ == "__main__":
    # Thay đường dẫn tới file video cabin xe hơi của bạn
    VIDEO_FILE =  "media.mp4"
    if Path(VIDEO_FILE).exists(): 
        transcript = process_driver_video(VIDEO_FILE)
    else:
        print(f"Vui lòng cung cấp file video hợp lệ tại đường dẫn: {VIDEO_FILE}")


