import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = r"C:\Users\a\OneDrive\Desktop\new deepfake\model\deepfake_model.h5"
model = load_model(MODEL_PATH)

# Image size expected by MobileNetV2
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    """Preprocess uploaded image for prediction."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image) / 255.0   # normalize [0,1]
    return np.expand_dims(image_array, axis=0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Preprocess
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        # Predict
        prediction = model.predict(processed_image)[0][0]

        # Real vs Fake probabilities
        fake_prob = float(prediction * 100)
        real_prob = float((1 - prediction) * 100)

        return jsonify({
            "real_probability": round(real_prob, 2),
            "fake_probability": round(fake_prob, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

