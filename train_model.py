import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==== CONFIG ====
DATASET_PATH = "shd_dataset"
IMG_SIZE = (150, 150)
EPOCHS = 25
BATCH_SIZE = 32
CORRUPT_SUFFIX = "_corrupt"
FEATURES_CSV = "image_features.csv"

# ==== Internal Feature Extraction ====
def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Rename unreadable image
            corrupt_path = os.path.splitext(image_path)[0] + CORRUPT_SUFFIX + os.path.splitext(image_path)[1]
            os.rename(image_path, corrupt_path)
            print(f"❌ Corrupt image renamed: {corrupt_path}")
            return None

        image = cv2.resize(image, IMG_SIZE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        brightness = np.mean(image_hsv[:, :, 2])
        contrast = np.std(image_rgb)
        saturation = np.mean(image_hsv[:, :, 1])
        sharpness = cv2.Laplacian(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

        reshaped = image_rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(reshaped)
        dominant_colors = kmeans.cluster_centers_.astype(int)

        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness": round(sharpness, 2),
            "dominant_color_1": dominant_colors[0].tolist(),
            "dominant_color_2": dominant_colors[1].tolist(),
            "dominant_color_3": dominant_colors[2].tolist()
        }
    except Exception as e:
        print(f"⚠️ Error in extract_features: {e}")
        return None

# ==== Load and process dataset ====
image_data = []
image_features = []

for filename in os.listdir(DATASET_PATH):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(DATASET_PATH, filename)
    features = extract_features(image_path)
    if features:
        features["image_name"] = filename
        image_features.append(features)

        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMG_SIZE)
            image = image / 255.0
            image_data.append(image)
        except Exception as e:
            print(f"⚠️ Failed to process image pixels for {filename}: {e}")

# ==== Convert to DataFrame ====
df = pd.DataFrame(image_features)

# ==== Generate Ratings ====
def generate_rating(row):
    return (
        row["brightness"] * 0.15 +
        row["contrast"] * 0.15 +
        row["saturation"] * 0.15 +
        row["sharpness"] * 0.15
    )

df["raw_rating"] = df.apply(generate_rating, axis=1)
scaler = MinMaxScaler(feature_range=(1, 10))
df["rating"] = scaler.fit_transform(df["raw_rating"].values.reshape(-1, 1)).flatten().round(2)

# ==== Save Features ====
df.to_csv(FEATURES_CSV, index=False)
print(f"📁 Features saved to {FEATURES_CSV}")

# ==== Prepare Data ====
X = np.array(image_data)
y = np.array(df["rating"])

if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("❌ No valid images available for training.")

# ==== CNN Model ====
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['mae'])

print("🚀 Starting model training...")
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
model.save("trained_model.h5")
print("✅ Model training complete. Saved as trained_model.h5")
