import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time
import io

# ==========================================
# REVISI LOGIKA BACKEND: MENAMBAHKAN BOUNDARIES VISUAL PADA CITRA
# ==========================================
def process_and_draw_boundaries(image_bytes):
    """
    Fungsi ini mensimulasikan deteksi wajah OpenCV (Haar Cascade) 
    dan menggambar bounding box serta persentase emosi langsung di atas gambar.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    h, w, _ = img.shape
    
    # Koordinat simulasi kotak wajah (Bounding Box)
    x1, y1 = int(w * 0.25), int(h * 0.2)
    x2, y2 = int(w * 0.75), int(h * 0.8)
    
    # Generasi probabilitas emosi (Simulasi model CNN Anda)
    emotions = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Disgust"]
    np.random.seed(len(image_bytes) % 1000)
    probs = np.random.dirichlet(np.ones(7), size=1)[0]
    result = {emotions[i]: float(probs[i]) for i in range(7)}
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    
    # Ambil emosi tertinggi untuk digambar di bounding box OpenCV
    top_emo = list(sorted_result.keys())[0]
    top_conf = sorted_result[top_emo] * 100
    
    # Canvas kloning untuk digambar kotak & teks
    annotated_img = img.copy()
    
    # Warna Bounding Box (Neon Cyan #38BDF8 -> BGR: 248, 189, 56)
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (248, 189, 56), 3)
    
    # Label teks overlay di atas kotak wajah
    label = f"{top_emo}: {top_conf:.1f}%"
    cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 189, 56), 2)
    
    # Crop bagian wajah asli saja untuk kolom sebelah kanan
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        face_crop = img

    return annotated_img, face_crop, sorted_result

# ==========================================
# 1. INITIALIZE SESSION STATE
# ==========================================
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ==========================================
# 2. CONFIGURATION & THEME INJECTION
# ==========================================
st.set_page_config(
    page_title="AI Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px);
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 35px;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .ai-card {
        background-color: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 10px;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed rgba(56, 189, 248, 0.3) !important;
        border-radius: 16px !important;
    }
    
    [data-testid="stCameraInput"] button {
        background: linear-gradient(90deg, #38BDF8 0%, #2563EB 100%) !important;
        color: white !important;
        border-radius: 12px !important;
    }
    
    .analyze-btn button {
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        width: 100% !important;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3) !important;
    }
    
    .cam-toggle-on button { background-color: #22C55E !important; color: white !important; border-radius: 12px !important; width: 100% !important; border: none !important; }
    .cam-toggle-off button { background-color: #EF4444 !important; color: white !important; border-radius: 12px !important; width: 100% !important; border: none !important; }
    
    /* Live HUD Overlay Sidebar Mini */
    .hud-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    
    .emotion-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 20px;
    }
    .emotion-item {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #2563EB;
        padding: 14px;
        border-radius: 0 12px 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 Sistem Informasi")
    st.markdown("""
    <div class="ai-card" style="padding: 15px; background: rgba(15, 23, 42, 0.4);">
        <p style='margin-bottom:8px; font-size:0.9rem;'><b>📌 INFO MODEL:</b></p>
        <ul style='margin-bottom:0; padding-left:20px; font-size:0.85rem; color:#94A3B8;'>
            <li>Convolutional Neural Network</li>
            <li>Backend: TensorFlow</li>
            <li>Face Tracker: OpenCV</li>
            <li>Input Target: 48x48 Pixels</li>
            <li>Output Layer: 7 Classes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Target Kelas Emosi")
    emotions_list = {
        "😊 Happy": "Bahagia", "😢 Sad": "Sedih", "😠 Angry": "Marah",
        "😨 Fear": "Takut", "😮 Surprise": "Terkejut", "😐 Neutral": "Datar",
        "🤢 Disgust": "Jijik"
    }
    for emo, desc in emotions_list.items():
        st.markdown(f"**{emo}** <span style='color:#64748B; font-size:0.85rem;'>({desc})</span>", unsafe_allow_html=True)

# ==========================================
# 4. HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 AI Facial Emotion Recognition</div>
    <div style="color: #94A3B8; margin-top:10px;">Sistem Kecerdasan Buatan tingkat lanjut untuk mendeteksi wajah secara real-time dengan Live HUD Boundaries overlay.</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([7, 3])

with col_left:
    st.markdown("""<div class="card-header">📥 Pilih Metode Input Citra Wajah</div>""", unsafe_allow_html=True)
    tab_upload, tab_camera = st.tabs(["📤 Unggah File Foto", "📷 Kamera Langsung (Live Cam)"])
    
    active_image_bytes = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Pilih file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_upload")
        if uploaded_file is not None:
            active_image_bytes = uploaded_file.read()
            
    with tab_camera:
        col_btn1, _ = st.columns([1, 2])
        with col_btn1:
            if not st.session_state.camera_active:
                st.markdown('<div class="cam-toggle-on">', unsafe_allow_html=True)
                if st.button("🟢 Aktifkan Kamera", key="act_cam"):
                    st.session_state.camera_active = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cam-toggle-off">', unsafe_allow_html=True)
                if st.button("🔴 Matikan Kamera", key="deact_cam"):
                    st.session_state.camera_active = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.camera_active:
            camera_file = st.camera_input("Ambil Foto", label_visibility="collapsed", key="cam_input")
            if camera_file is not None:
                active_image_bytes = camera_file.read()
        else:
            st.markdown('<div style="background-color: rgba(30, 41, 59, 0.4); padding: 30px; border-radius: 12px; text-align: center; margin-top: 15px; color: #64748B;">Status Kamera: <b>OFF</b></div>', unsafe_allow_html=True)

    # PEMROSESAN UTAMA DENGAN LIVE HUD BOUNDARIES
    if active_image_bytes is not None:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        col_img1, col_img2 = st.columns(2)
        
        # Ekstraksi frame ter-overlay box dan persentase emosi mikro
        annotated_img, face_crop, emotion_predictions = process_and_draw_boundaries(active_image_bytes)
        
        with col_img1:
            st.markdown('<div class="ai-card"><div class="card-header">📷 Live AI Tracker Frame (Boundaries)</div>', unsafe_allow_html=True)
            # Menampilkan gambar asli yang sudah digambari kotak wajah & persentase emosi oleh OpenCV
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_img2:
            st.markdown('<div class="ai-card"><div class="card-header">🎯 Wajah Terisolasi & Live HUD</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Tampilan HUD mini berisi daftar persentase boundaries emosi langsung di dalam card wajah
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            emoji_map = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Fear": "😨", "Surprise": "😮", "Neutral": "😐", "Disgust": "🤢"}
            
            # Tampilkan 3 emosi teratas langsung sebagai HUD boundaries di bawah potongan wajah
            for index, (emo_name, prob_val) in enumerate(list(emotion_predictions.items())[:3]):
                pct = prob_val * 100
                st.markdown(f"""
                <div class="hud-metric">
                    <span>{emoji_map.get(emo_name, '')} <b>{emo_name}</b></span>
                    <span style="color:#38BDF8; font-weight:700;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ==========================================
        # 5. TOMBOL ANALISIS & PROBABILITY CHART
        # ==========================================
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        if st.button("🔍 Analisis Ekspresi Wajah", key="deep_analyze"):
            with st.spinner("AI sedang menghitung matriks probabilitas..."):
                time.sleep(0.3) # Respons kilat premium
                
            top_emotion = list(emotion_predictions.keys())[0]
            top_confidence = emotion_predictions[top_emotion] * 100
            
            color_theme = "#22C55E" if top_confidence >= 80 else ("#F59E0B" if top_confidence >= 50 else "#EF4444")
            
            st.markdown(f"""
            <div class="ai-card" style="border-left: 6px solid {color_theme}; margin-top:20px;">
                <div class="card-header">🏆 Hasil Deteksi Utama</div>
                <div class="emotion-badge">{emoji_map.get(top_emotion, '👤')} {top_emotion.upper()}</div>
                <div style="color:#94A3B8; font-size:0.95rem;">Confidence Score: <b style="color:{color_theme};">{top_confidence:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<style>.stProgress > div > div > div > div {{ background-color: {color_theme} !important; }}</style>", unsafe_allow_html=True)
            st.progress(int(top_confidence))

            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="card-header">📊 Probabilitas Seluruh Emosi</div>', unsafe_allow_html=True)
            chart_data = {f"{emoji_map.get(k, '')} {k}": v for k, v in emotion_predictions.items()}
            st.bar_chart(chart_data, horizontal=True, color="#7C3AED")
        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # ==========================================
    # 6. STATISTIK SISTEM
    # ==========================================
    st.markdown("""
    <div class="ai-card">
        <div class="card-header">📈 Statistik Sistem</div>
        <table style="width:100%; border-collapse: collapse; font-size:0.9rem;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding: 10px 0; color:#94A3B8;">Total Kelas</td><td style="text-align:right; font-weight:600; color:#38BDF8;">7 Kategori</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding: 10px 0; color:#94A3B8;">Resolusi Input</td><td style="text-align:right; font-weight:600; color:#38BDF8;">48x48 Matrix</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding: 10px 0; color:#94A3B8;">Framework</td><td style="text-align:right; font-weight:600; color:#38BDF8;">TensorFlow</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding: 10px 0; color:#94A3B8;">Face Detector</td><td style="text-align:right; font-weight:600; color:#38BDF8;">OpenCV Engine</td></tr>
            <tr><td style="padding: 10px 0; color:#94A3B8;">Tipe Model</td><td style="text-align:right; font-weight:600; color:#38BDF8;">CNN Deep Nets</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. INFORMASI SPEKTRUM EMOSI
# ==========================================
st.markdown("---")
st.markdown('<div class="card-header">📚 Penjelasan Spektrum Emosi Wajah</div>', unsafe_allow_html=True)
st.markdown("""
<div class="emotion-grid">
    <div class="emotion-item" style="border-left-color: #22C55E;"><b style="color:#22C55E;">😊 Happy</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Kebahagiaan atau kepuasan sosial.</p></div>
    <div class="emotion-item" style="border-left-color: #3B82F6;"><b style="color:#3B82F6;">😢 Sad</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Kekecewaan atau penurunan mood pasif.</p></div>
    <div class="emotion-item" style="border-left-color: #EF4444;"><b style="color:#EF4444;">😠 Angry</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Kemarahan, resistensi, atau frustrasi.</p></div>
    <div class="emotion-item" style="border-left-color: #F59E0B;"><b style="color:#F59E0B;">😨 Fear</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Rasa takut atau respon proteksi diri mendadak.</p></div>
    <div class="emotion-item" style="border-left-color: #A855F7;"><b style="color:#A855F7;">😮 Surprise</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Keterkejutan atas stimulus tak terduga.</p></div>
    <div class="emotion-item" style="border-left-color: #64748B;"><b style="color:#64748B;">😐 Neutral</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Ekspresi normal tanpa emosi dominan.</p></div>
    <div class="emotion-item" style="border-left-color: #EC4899;"><b style="color:#EC4899;">🤢 Disgust</b><p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">Rasa jijik atau penolakan tinggi.</p></div>
</div>
<div style="text-align:center; padding: 40px 0 20px 0; font-size:0.8rem; color:#475569;">
    Enterprise Engine AI Engine Core v2.4.0 • Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
