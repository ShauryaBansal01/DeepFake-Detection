import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Parameters
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 10

# Step 1: Get all video paths
def get_all_video_paths():
    base_dirs = [
        "dataset/Celeb-DF",
        "dataset/DFDC",
        "dataset/FF++"
    ]

    real_paths = []
    fake_paths = []

    for base in base_dirs:
        real_dir = os.path.join(base, "real")
        fake_dir = os.path.join(base, "fake")

        if os.path.exists(real_dir):
            real_paths += [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".mp4")]
        else:
            print(f"⚠️ Real dir not found: {real_dir}")

        if os.path.exists(fake_dir):
            fake_paths += [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".mp4")]
        else:
            print(f"⚠️ Fake dir not found: {fake_dir}")

    print(f"✅ Found {len(real_paths)} real videos and {len(fake_paths)} fake videos")
    return real_paths, fake_paths

# Step 2: Extract frames from a single video
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
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    frames_array = np.array(frames)
    if frames_array.shape == (FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3):
        return frames_array
    else:
        return None

# Step 3: Prepare dataset
def prepare_dataset(real_paths, fake_paths):
    X, y = [], []

    print("\n🎞️  Processing real videos...")
    for path in tqdm(real_paths):
        frames = extract_frames(path)
        if frames is not None:
            X.append(frames)
            y.append(0)

    print("\n🎭 Processing fake videos...")
    for path in tqdm(fake_paths):
        frames = extract_frames(path)
        if frames is not None:
            X.append(frames)
            y.append(1)

    X = np.array(X)
    y = np.array(y)
    print(f"\n📦 Dataset shape: {X.shape}, Labels shape: {y.shape}")
    return X, y

# Step 4: Build model
def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main
if __name__ == "__main__":
    print("🚀 Starting deepfake detection training pipeline...\n")
    print("🔍 Scanning dataset folders...")
    real_video_paths, fake_video_paths = get_all_video_paths()

    if not real_video_paths or not fake_video_paths:
        print("❌ No videos found. Please check your dataset folder paths.")
        exit()

    print("\n📦 Loading and processing video data...")
    X, y = prepare_dataset(real_video_paths, fake_video_paths)
    print(f"\n✅ Dataset prepared: {X.shape[0]} samples, each with {FRAMES_PER_VIDEO} frames of size {FRAME_WIDTH}x{FRAME_HEIGHT}")

    # Debug input
    print(f"\n🧪 Sample input shape: {X[0].shape}")
    print(f"🧪 Model input expected shape: ({FRAMES_PER_VIDEO}, {FRAME_HEIGHT}, {FRAME_WIDTH}, 3)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🧠 Building model...")
    model = build_model()

    print("\n🏋️‍♀️ Starting training...\n")
    model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

    print("\n✅ Training complete! Saving model as `deepfake_model.h5`")
    model.save("deepfake_model.h5")
