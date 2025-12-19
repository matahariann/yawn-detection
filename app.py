import os
import tempfile
from pathlib import Path
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2

st.set_page_config(
    page_title="Yawn Detection AI",
    page_icon="🥱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS untuk tampilan modern
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 2rem 3rem;
    }
    
    h1 {
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        padding: 0 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
    }
    
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        background-color: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background-color: #f1f5f9;
    }
    
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-active {
        background: #dcfce7;
        color: #16a34a;
    }
    
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .stProgress > div > div {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("# 🥱 Yawn Detection AI")
st.markdown('<p class="subtitle">Deteksi menguap secara real-time menggunakan YOLOv8</p>', unsafe_allow_html=True)

# ---------------------------
# Utils
# ---------------------------
@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)

def infer_image_bgr(model: YOLO, frame_bgr: np.ndarray, imgsz: int, conf: float) -> np.ndarray:
    results = model.predict(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)
    annotated = results[0].plot()
    return annotated

def save_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def process_video(model: YOLO, in_path: str, out_path: str, imgsz: int, conf: float) -> None:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Video tidak bisa dibuka. Pastikan format video valid.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codecs = [
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
    ]
    
    writer = None
    actual_output = None
    
    for codec, ext in codecs:
        try:
            temp_output = out_path.replace(".mp4", ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))
            if test_writer.isOpened():
                writer = test_writer
                actual_output = temp_output
                break
            else:
                test_writer.release()
        except:
            continue
    
    if writer is None:
        raise RuntimeError("Tidak bisa membuat video writer dengan codec yang tersedia")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    progress = st.progress(0)
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = infer_image_bgr(model, frame, imgsz=imgsz, conf=conf)
        writer.write(annotated)

        processed += 1
        if frame_count > 0:
            progress.progress(min(processed / frame_count, 1.0))

    cap.release()
    writer.release()
    progress.empty()
    
    if actual_output != out_path:
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            st.info("Mengkonversi ke MP4...")
            subprocess.run([
                "ffmpeg", "-i", actual_output,
                "-vcodec", "libx264",
                "-acodec", "aac",
                "-y",
                out_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            os.remove(actual_output)
        except:
            st.warning(f"Video disimpan sebagai {actual_output}. Install FFmpeg untuk konversi otomatis ke MP4.")
            if os.path.exists(actual_output):
                import shutil
                shutil.copy(actual_output, out_path)

# ---------------------------
# Load model
# ---------------------------
default_weights = "best_model.pt"
conf = 0.25
imgsz = 640

if not Path(default_weights).exists():
    st.error("⚠️ File best_model.pt tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py.")
    st.stop()

model = load_model(default_weights)

# ---------------------------
# WebRTC Video Processor
# ---------------------------
class YawnDetectionProcessor:
    def __init__(self):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        annotated = infer_image_bgr(self.model, img, imgsz=self.imgsz, conf=self.conf)
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------------------------
# Tabs
# ---------------------------
tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Gambar", "🎞️ Video", "📷 Webcam"])

# ===== Image Tab =====
with tab_img:
    st.markdown("### Upload Gambar")
    st.markdown("Upload gambar untuk mendeteksi apakah seseorang sedang menguap")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    up_img = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "webp"],
        key="img",
        help="Format: JPG, PNG, WEBP"
    )
    
    if up_img is not None:
        file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Gagal membaca gambar. Coba format lain (jpg/png).")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("**📥 Gambar Asli**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col2:
                st.markdown("**🎯 Hasil Deteksi**")
                with st.spinner("Memproses..."):
                    pred_bgr = infer_image_bgr(model, img_bgr, imgsz=imgsz, conf=conf)
                st.image(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

# ===== Video Tab =====
with tab_vid:
    st.markdown("### Upload Video")
    st.markdown("Upload video untuk mendeteksi menguap secara frame-by-frame")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    up_vid = st.file_uploader(
        "Pilih video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid",
        help="Format: MP4, MOV, AVI, MKV, WEBM"
    )
    
    if up_vid is not None:
        in_path = save_uploaded_file(up_vid, suffix="." + up_vid.name.split(".")[-1])
        out_path = os.path.join(tempfile.gettempdir(), f"pred_{Path(in_path).stem}.mp4")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📥 Video Input**")
        st.video(in_path)

        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            process_btn = st.button("🚀 Proses Video", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("⏳ Memproses video... Mohon tunggu"):
                try:
                    process_video(model, in_path=in_path, out_path=out_path, imgsz=imgsz, conf=conf)
                    st.success("✅ Selesai! Video berhasil diproses")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**🎯 Hasil Deteksi Video**")
                    st.video(out_path)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Hasil Video",
                            f,
                            file_name="yawn_detection_result.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"❌ Gagal memproses video: {e}")

# ===== Webcam Tab =====
with tab_cam:
    st.markdown("### Deteksi Real-time dengan Webcam")
    st.markdown("Gunakan webcam untuk mendeteksi menguap secara langsung")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("💡 **Cara Menggunakan:**\n"
            "1. Klik tombol **START** di bawah ini\n"
            "2. Browser akan meminta izin akses webcam - klik **Allow/Izinkan**\n"
            "3. Webcam akan mulai mendeteksi menguap secara real-time\n"
            "4. Klik **STOP** untuk menghentikan")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # RTC Configuration for better connectivity
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="yawn-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YawnDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if webrtc_ctx.state.playing:
        st.markdown('<span class="status-badge status-active">🟢 Webcam Aktif</span>', unsafe_allow_html=True)
    else:
        st.markdown("💡 Klik **START** untuk memulai deteksi webcam")

st.divider()