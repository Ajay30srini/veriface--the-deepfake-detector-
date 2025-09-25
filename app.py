import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import gdown

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Model Setup
# ------------------------------
# Use relative path so it works on Render (Linux) and locally
MODEL_PATH = os.path.join("model", "deepfake_model.h5")

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1ZEbB4xDMDmAfzWJG3WS5zlxucj6oohvK"
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# Load model
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle uploaded file
        file = request.files["file"]
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        result = "FAKE" if prediction[0][0] > 0.5 else "REAL"

        return render_template("index.html", prediction=result)

    return render_template("index.html")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
