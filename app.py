import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from flask import Flask, request, jsonify, render_template
from sklearn.cluster import KMeans

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
DASHBOARD_FOLDER = "static/dashboards"
IMG_SIZE = (150, 150)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

MODEL_PATH = "trained_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.resize(image, IMG_SIZE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        brightness = float(np.mean(hsv[:, :, 2]))
        contrast = float(np.std(rgb))
        saturation = float(np.mean(hsv[:, :, 1]))
        sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

        reshaped = rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(reshaped)
        colors = kmeans.cluster_centers_.astype(int)

        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness": round(sharpness, 2),
            "dominant_color_1": colors[0].tolist(),
            "dominant_color_2": colors[1].tolist(),
            "dominant_color_3": colors[2].tolist()
        }
    except Exception as e:
        print(f"Feature error: {e}")
        return None

def analyze_decor(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = len([c for c in contours if cv2.contourArea(c) > 500])
        return count
    except Exception as e:
        print(f"Decor analysis error: {e}")
        return 0

def generate_dashboard(features, save_path):
    try:
        plt.switch_backend("Agg")
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2], polar=True)

        # --- Furniture & Decor Impact
        decor_items = features.get("decor_count", 0)
        furniture_items = features.get("furniture_count", 0)
        ax1.bar(["Furniture", "Decor"], [furniture_items, decor_items], color=["#8ecae6", "#ffb703"])
        ax1.set_title("Furniture & Decor Impact")
        ax1.set_ylabel("Count")

        # --- Dominant Color Bar
        dom_colors = [
            features["dominant_color_1"],
            features["dominant_color_2"],
            features["dominant_color_3"]
        ]
        ax2.bar(["Color 1", "Color 2", "Color 3"], [1, 1, 1], color=[np.array(c)/255 for c in dom_colors])
        ax2.set_title("Dominant Colors")
        ax2.set_yticks([])

        # --- Aesthetic Balance Radar Chart
        categories = ["Brightness", "Contrast", "Saturation", "Sharpness"]
        ideal = [150, 50, 60, 50]
        actual = [
            features["brightness"],
            features["contrast"],
            features["saturation"],
            features["sharpness"]
        ]
        ideal_norm = [1 for _ in ideal]
        actual_norm = [
            min(actual[0] / 180, 1),
            min(actual[1] / 70, 1),
            min(actual[2] / 80, 1),
            min(actual[3] / 80, 1),
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        actual_norm += actual_norm[:1]
        ideal_norm += ideal_norm[:1]
        angles += angles[:1]

        ax3.plot(angles, ideal_norm, linestyle='dashed', color='gray', label="Ideal")
        ax3.plot(angles, actual_norm, color='teal', label="Your Room")
        ax3.fill(angles, actual_norm, color='teal', alpha=0.25)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_yticklabels([])
        ax3.set_title("Aesthetic Balance Radar", pad=20)
        ax3.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"Dashboard error: {e}")




def calculate_aesthetic_score(features):
    score = 0
    if 120 <= features["brightness"] <= 180: score += 2
    elif 100 <= features["brightness"] <= 200: score += 1

    if features["contrast"] > 50: score += 2
    elif features["contrast"] > 35: score += 1

    if features["saturation"] > 60: score += 2
    elif features["saturation"] > 40: score += 1

    if features["sharpness"] > 40: score += 2
    elif features["sharpness"] > 25: score += 1

    return round(min(score / 8 * 10, 2), 2)

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        timestamp = int(time.time())
        ext = os.path.splitext(file.filename)[1]
        filename = f"image_{timestamp}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Image unreadable"}), 400

        resized = cv2.resize(image, IMG_SIZE).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(resized, axis=0)

        if model is None:
            return jsonify({"error": "Model not available"}), 500

        features = extract_features(file_path)
        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        decor_count = analyze_decor(file_path)
        features["decor_count"] = decor_count

        rule_rating = calculate_aesthetic_score(features)
        model_rating = float(np.clip(model.predict(input_tensor)[0][0], 1, 10))
        final_rating = round((rule_rating + model_rating) / 2, 2)

        features["rating"] = final_rating
        features["rule_rating"] = rule_rating
        features["ml_rating"] = round(model_rating, 2)

        recs = []

        # Brightness
        if 120 <= features["brightness"] <= 180:
            recs.append("✅ Brightness is well-balanced — no changes needed.")
        elif features["brightness"] < 100:
            recs.append("🔆 The room appears dim. Try adding warm white ceiling lights or increasing daylight.")
        elif features["brightness"] > 220:
            recs.append("⚠️ Image is overexposed — consider reducing direct lighting or closing bright curtains.")

        # Contrast
        if features["contrast"] < 30:
            recs.append("🌗 Low contrast detected — try adding darker accents or textures for visual depth.")
        elif features["contrast"] > 70:
            recs.append("🌓 High contrast — consider softening transitions with neutral tones.")

        # Saturation
        if features["saturation"] < 40:
            recs.append("🎨 Add vibrant decor like cushions, plants, or colorful paintings.")
        elif features["saturation"] > 70:
            recs.append("🌈 Too much saturation — balance with natural or neutral elements.")

        # Sharpness
        if features["sharpness"] < 25:
            recs.append("📷 The image is a bit blurry — try better lighting or a sharper camera angle.")
        elif features["sharpness"] > 80:
            recs.append("🔍 Sharpness is great — details are very clear.")

        # Decor count
        if decor_count < 10:
            recs.append("🪑 Room feels a bit empty — consider adding wall art, cushions or a floor lamp.")
        elif decor_count > 30:
            recs.append("🧺 Too many items — declutter to create a more open and peaceful space.")

        # Overall
        if final_rating < 5:
            recs.append("🛋️ Consider reorganizing layout or using symmetry for aesthetic balance.")
        elif final_rating >= 8:
            recs.append("🏆 Excellent visual balance — beautiful interior!")

        if not recs:
            recs.append("👍 The room looks balanced and visually appealing.")

        features["recommendations"] = recs

        dashboard_name = f"dashboard_{timestamp}.png"
        dashboard_path = os.path.join(DASHBOARD_FOLDER, dashboard_name)
        generate_dashboard(features, dashboard_path)
        features["dashboard"] = dashboard_path.replace("\\", "/") + f"?t={timestamp}"

        print("🎯 Final features + rating:", features)
        return jsonify(features)

    except Exception as e:
        print(f"🔥 Prediction error: {e}")
        return jsonify({"error": "Server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)
