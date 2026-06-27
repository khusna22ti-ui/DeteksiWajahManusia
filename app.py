import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "emotion_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# LABEL KELAS (HARUS SESUAI TRAINING)
# ==========================
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================
# CONFIG STREAMLIT
# ==========================
st.set_page_config(
    page_title="Deteksi Ekspresi Wajah",
    page_icon="😊",
    layout="centered"
)

st.title("Sistem Deteksi Ekspresi Wajah")
st.write("Upload gambar wajah untuk mendeteksi ekspresi.")

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# FACE DETECTION (OPSIONAL TAPI DISARANKAN)
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def detect_face(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return image  # kalau tidak ada wajah, pakai original

    # ambil wajah pertama
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)

# ==========================
# PREDIKSI
# ==========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar Asli", use_container_width=True)

    if st.button("Prediksi Ekspresi"):

        # ==========================
        # Crop wajah (jika ada)
        # ==========================
        face_image = detect_face(image)

        st.image(face_image, caption="Wajah Terdeteksi", use_container_width=True)

        # ==========================
        # PREPROCESSING
        # ==========================
        img = face_image.convert("L")      # grayscale
        img = img.resize((48, 48))         # resize sesuai model

        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # ==========================
        # PREDIKSI MODEL
        # ==========================
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[0][predicted_index]) * 100

        # ==========================
        # OUTPUT HASIL
        # ==========================
        st.subheader("Hasil Prediksi")

        st.success(f"Ekspresi: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        if confidence < 40:
            st.warning("Model kurang yakin dengan hasil ini.")

        # ==========================
        # VISUALISASI PROBABILITAS
        # ==========================
        prob_df = pd.DataFrame({
            "Ekspresi": class_names,
            "Probabilitas": prediction[0]
        })

        st.subheader("Probabilitas Setiap Kelas")
        st.bar_chart(prob_df.set_index("Ekspresi"))