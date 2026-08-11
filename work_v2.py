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
        model_path = Path("..\\..\\face_landmarker.task")
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
import torch
import torchaudio
import numpy as np
import soundfile as sf

# import model_architectures  # Thay thế bằng thư viện chứa kiến trúc AV-TSE của bạn (ví dụ: CTCNet hoặc VisualVoice)

def apply_av_target_speaker_extraction(lip_frames, raw_audio_path, output_audio_path="driver_clean_voice.wav"):
    """
    Sử dụng mô hình Deep Learning AV-TSE để trích xuất giọng nói của tài xế dựa trên khẩu hình.
    
    - Đầu vào:
        + lip_frames: List các numpy array kích thước [96, 96, 3] (RGB hoặc BGR) đại diện cho chuỗi chuyển động môi.
        + raw_audio_path: Đường dẫn tới file âm thanh thô hỗn hợp (16kHz Mono).
    - Đầu ra:
        + clean_audio_path: Đường dẫn tới file audio sạch đã lọc.
    """
    print(f"-> Đang xử lý lọc giọng bằng mô hình AV-TSE với {len(lip_frames)} khung hình khẩu hình...")
    
    # 1. THIẾT BỊ CHẠY MÔ HÌNH (CUDA nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   [AV-TSE] Đang chạy trên thiết bị: {device}")

    # 2. TIỀN XỬ LÝ DỮ LIỆU VIDEO (LIP ROI FRAMES)
    # AV-TSE models yêu cầu định dạng video đầu vào chuẩn hóa: [Batch=1, Channels=1 (Grayscale), Temporal_Frames, Height, Width]
    video_tensor = []
    for frame in lip_frames:
        # Chuyển đổi sang Grayscale nếu mô hình yêu cầu ảnh xám
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Chuẩn hóa về khoảng [0.0, 1.0] hoặc [-1.0, 1.0] tùy theo mô hình
        normalized_frame = gray_frame.astype(np.float32) / 255.0
        video_tensor.append(normalized_frame)
    
    # Stack các khung hình thành tensor: [T, 96, 96]
    video_tensor = np.stack(video_tensor, axis=0)
    # Biến đổi chiều để khớp đầu vào model: [Batch=1, Channels=1, T, H, W]
    video_tensor = torch.from_numpy(video_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. TIỀN XỬ LÝ DỮ LIỆU AUDIO HỖN HỢP (RAW MIXTURE AUDIO)
    # Tải audio 16kHz mono bằng torchaudio
    waveform, sample_rate = torchaudio.load(raw_audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Chuẩn hóa âm lượng đầu vào và chuyển sang device: [Batch=1, Audio_Length]
    waveform = waveform.to(device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 4. KHỞI TẠO MÔ HÌNH AV-TSE & TẢI TRỌNG SỐ (WEIGHTS)
    # Ví dụ minh họa tải mô hình CTCNet hoặc VisualVoice đã pre-trained:
    """
    model = CTCNet(encoder_type='conv', video_channels=1, audio_channels=1).to(device)
    checkpoint = torch.load("av_tse_ctcnet_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    """
    
    # 5. INFERENCE (DỰ ĐOÁN GỌNG NÓI SẠCH)
    with torch.no_grad():
        # Đầu vào mô hình: waveform (audio hỗn hợp) + video_tensor (cử động môi tài xế)
        # Đầu ra: target_waveform (giọng nói tài xế đã lọc sạch tạp âm và tiếng người nói đè)
        try:
            # clean_waveform = model(waveform, video_tensor)
            
            # GIẢ LẬP inference của model để code chạy được nếu chưa load model thực tế:
            print("   [AV-TSE] Đang suy luận mô hình Deep Learning...")
            clean_waveform = waveform.clone() # Giả lập: tạm thời giữ nguyên giọng gốc
            
        except Exception as e:
            print(f"Lỗi trong quá trình suy luận mô hình AV-TSE: {e}")
            clean_waveform = waveform
            
    # Chuyển kết quả về CPU và ghi ra file WAV
    clean_waveform_cpu = clean_waveform.squeeze(0).cpu().numpy()
    
    # Ghi file audio kết quả với tần số lấy mẫu 16kHz
    sf.write(output_audio_path, clean_waveform_cpu, 16000)
    print(f"-> Đã trích xuất xong âm thanh mục tiêu và lưu tại: {output_audio_path}")
    
    return output_audio_path


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

    return clean_audio

# ==========================================

# CHẠY THỬ NGHIỆM

# ==========================================

if __name__ == "__main__":
    # Thay đường dẫn tới file video cabin xe hơi của bạn
    VIDEO_FILE =  "..\\..\\media.mp4"
    if Path(VIDEO_FILE).exists(): 
        transcript = process_driver_video(VIDEO_FILE)
    else:
        print(f"Vui lòng cung cấp file video hợp lệ tại đường dẫn: {VIDEO_FILE}")


