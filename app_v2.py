import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Parameters
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 10
EPOCHS = 20
BATCH_SIZE = 4

# Step 1: Get all video paths
def get_all_video_paths():
    base_dirs = ["dataset/Celeb-DF", "dataset/DFDC", "dataset/FF++"]
    real_paths, fake_paths = [], []

    for base in base_dirs:
        real_dir = os.path.join(base, "real")
        fake_dir = os.path.join(base, "fake")

        if os.path.exists(real_dir):
            real_paths += [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".mp4")]
        if os.path.exists(fake_dir):
            fake_paths += [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".mp4")]

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
        frame = preprocess_input(frame.astype(np.float32))
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

    print("\n🧹 Cleaning data...")
    X_cleaned, y_cleaned = [], []
    for i in range(len(X)):
        if X[i] is not None and X[i].shape == (FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3):
            X_cleaned.append(X[i])
            y_cleaned.append(y[i])

    X = np.array(X_cleaned)
    y = np.array(y_cleaned)
    print(f"📦 Dataset shape: {X.shape}, Labels shape: {y.shape}")
    return X, y

# Step 4: Build model with MobileNetV2 base and Bidirectional LSTM
def build_model():
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    base_model.trainable = False

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(64)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main
if __name__ == "__main__":
    print("🚀 Starting deepfake detection training pipeline...\n")

    real_video_paths, fake_video_paths = get_all_video_paths()
    if not real_video_paths or not fake_video_paths:
        print("❌ No videos found. Please check dataset paths.")
        exit()

    print("\n📦 Loading and processing video data...")
    X, y = prepare_dataset(real_video_paths, fake_video_paths)

    print("\n🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🧠 Building model...")
    model = build_model()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print("\n🏋️ Training model...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    print("\n✅ Training complete! Saving model as `deepfake_model.h5`")
    model.save("deepfake_model_v2.h5")