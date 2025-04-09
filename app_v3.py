import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Params
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20
BATCH_SIZE = 4
EPOCHS = 20

# Dataset paths
DATASETS = ["dataset/Celeb-DF", "dataset/DFDC", "dataset/FF++"]

# 🔁 Custom Data Generator
class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.on_epoch_end()

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.video_paths[k] for k in batch_idx]
        batch_labels = [self.labels[k] for k in batch_idx]
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y

    def __data_generation(self, batch_paths, batch_labels):
        X = np.zeros((self.batch_size, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
        y = np.array(batch_labels, dtype=np.float32)

        for i, path in enumerate(batch_paths):
            frames = extract_frames(path)
            if frames is not None:
                X[i] = frames
        return X, y

# 🎞 Frame extractor
def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total < num_frames or total == 0:
        cap.release()
        return np.zeros((num_frames, FRAME_HEIGHT, FRAME_WIDTH, 3))

    interval = total // num_frames
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)

    cap.release()
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))
    return np.array(frames)

# 📦 Load video paths
def load_all_video_paths():
    real_paths, fake_paths = [], []
    for base in DATASETS:
        real_dir = os.path.join(base, "real")
        fake_dir = os.path.join(base, "fake")
        if os.path.exists(real_dir):
            real_paths += [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".mp4")]
        if os.path.exists(fake_dir):
            fake_paths += [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".mp4")]
    print(f"✅ {len(real_paths)} real | {len(fake_paths)} fake")
    return real_paths, fake_paths

# 🧠 Build model
def build_model():
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
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

# 🚀 Main Training
if __name__ == "__main__":
    real_paths, fake_paths = load_all_video_paths()
    video_paths = real_paths + fake_paths
    labels = [0]*len(real_paths) + [1]*len(fake_paths)

    X_train, X_val, y_train, y_val = train_test_split(video_paths, labels, test_size=0.2, random_state=42)

    train_gen = VideoDataGenerator(X_train, y_train, BATCH_SIZE, is_training=True)
    val_gen = VideoDataGenerator(X_val, y_val, BATCH_SIZE, is_training=False)

    model = build_model()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ModelCheckpoint("best_deepfake_model.h5", save_best_only=True)
    ]

    print("🏋️ Training model...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    print("✅ Done! Model saved to best_deepfake_model.h5")
