# 😊 Human Facial Emotion Detection using Computer Vision

This project is a human facial emotion detection system powered by a deep learning model built with TensorFlow/Keras. Instead of using a webcam, users can **upload an image**, and the system will analyze the face and predict the detected emotion, such as **Happy, Sad, Angry, Fear, Surprise, Disgust,** or **Neutral**.

# 🚀 Features

📷 Upload an image for facial emotion detection. <br/>
😊 Detects multiple facial expressions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral). <br/>
🖥️ Interactive and user-friendly interface built with Streamlit. <br/>
🤖 Deep learning model implemented in `.h5` format (CNN). <br/>
⚡ Fast and easy emotion prediction from uploaded images. <br/>

# 🧠 Technologies Used

* **Frontend & Backend:** Streamlit (Python) <br/>
* **Machine Learning:** TensorFlow / Keras <br/>
* **Computer Vision:** OpenCV <br/>
* **Image Processing:** NumPy <br/>

# 🗂️ Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset, a widely used benchmark for facial emotion recognition. The dataset consists of **35,887 grayscale facial images** with a resolution of **48×48 pixels**, categorized into seven emotion classes:

* 😠 Angry
* 🤢 Disgust
* 😨 Fear
* 😊 Happy
* 😢 Sad
* 😲 Surprise
* 😐 Neutral

The FER2013 dataset is publicly available on Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

## 📁 Project Structure

```text
emotion-detection
├── app.py                     # Main Streamlit application
├── Model                      # Model training script
├── Dataset                    # Dataset
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## ⚙️ Setup & Run

1. Clone this repository:

```bash
git clone https://github.com/khusna22ti/SistemDeteksiEmosiWajah.git
cd emotion-detection
```

2. (Optional) Train the model:

```bash
python model\training.py
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open your browser. Streamlit will automatically launch the application, usually at:

```
http://localhost:8501
```

5. Upload a facial image through the Streamlit interface and the system will predict the detected emotion.

---

Made with 💧 by Khusna Billa
