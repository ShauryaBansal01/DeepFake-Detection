import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

# Parameters (must match training)
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20

# Load model
model = load_model("deepfake_model_v2.h5")

# Frame extractor
def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames < num_frames or total_frames == 0:
        cap.release()
        return None

    interval = total_frames // num_frames

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = preprocess_input(frame.astype(np.float32))  # MobileNetV2 preprocessing
        frames.append(frame)

    cap.release()
    frames_array = np.array(frames)
    return frames_array if frames_array.shape == (FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3) else None

# Predict
def predict_video(video_path):
    frames = extract_frames(video_path)
    if frames is None:
        print("❌ Could not extract valid frames.")
        return

    input_data = np.expand_dims(frames, axis=0)  # Shape: (1, 10, 128, 128, 3)
    prediction = model.predict(input_data)[0][0]
    label = "FAKE" if prediction >= 0.5 else "REAL"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    print(f"🎬 Prediction: {label} ({confidence * 100:.2f}% confidence)")

# File picker
def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if video_path:
        print(f"\n📂 Selected video: {video_path}")
        predict_video(video_path)

# Run
if __name__ == "__main__":
    print("🎥 Deepfake Detector - Video Prediction\n")
    select_video()
