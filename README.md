# ✋ Air Handwritten Digit Recognition

A real-time Computer Vision and Deep Learning project that recognizes digits drawn in the air using finger movements.

The system combines **MediaPipe Hand Tracking**, **OpenCV**, and a **Convolutional Neural Network (CNN)** trained on the MNIST dataset to capture finger trajectories, generate digit images, and predict the corresponding digit in real time.

---

## 📌 Overview

Traditional digit recognition systems require users to write on paper or touchscreens. This project enables users to draw digits freely in the air using their index finger while a webcam captures the motion.

The application tracks the fingertip, creates a virtual canvas, preprocesses the drawing, and uses a trained CNN model to classify the digit instantly.

---

## 🚀 Features

✅ Real-time webcam-based digit recognition

✅ Hand and finger tracking using MediaPipe

✅ Air-writing using fingertip movement

✅ CNN trained on the MNIST dataset

✅ Digit prediction with confidence score

✅ Motion smoothing for cleaner drawings

✅ Canvas reset functionality

✅ Fast and lightweight inference

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection

* MNIST handwritten digit dataset
* 70,000 labeled digit images (0–9)

### 2. Preprocessing

* Grayscale conversion
* Image resizing (28×28)
* Normalization
* Noise reduction

### 3. Model Training

* Convolutional Neural Network (CNN)
* Multiple convolution and pooling layers
* Softmax output layer for digit classification

### 4. Real-Time Inference

* Hand tracking using MediaPipe
* Fingertip trajectory capture
* Canvas generation
* Digit prediction using trained CNN model

---

## 📊 Model Performance

| Metric           | Value           |
| ---------------- | --------------- |
| Dataset          | MNIST           |
| Training Samples | 60,000          |
| Test Samples     | 10,000          |
| Accuracy         | 98%+            |
| Classes          | 10 Digits (0–9) |

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning & Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* MediaPipe

### Data Processing

* NumPy

---

## 📂 Project Structure

```text
Air-Handwritten-Digit-Recognition/
│
├── model/
│   └── digit_model.h5
│
├── dataset/
│   └── MNIST
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/BinduPranaya/Air-Handwritten-Digit-Recognition.git
cd Air-Handwritten-Digit-Recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## 🎮 Controls

| Key | Action           |
| --- | ---------------- |
| C   | Clear Canvas     |
| Q   | Quit Application |

---

## 📈 Future Improvements

* Support alphabets and symbols
* Multi-digit recognition
* Gesture-based controls
* Mobile deployment
* Improved deep learning architectures

---

## 👨‍💻 Author

**Bindu Pranaya Mogilicherla**

* GitHub: https://github.com/BinduPranaya
* LinkedIn: https://linkedin.com/in/mogilicherla-bindu-pranaya-425131325

---

⭐ If you found this project useful, consider giving it a star.
