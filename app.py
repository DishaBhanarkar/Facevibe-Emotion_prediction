from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# ✅ CREATE APP FIRST
app = Flask(__name__)
CORS(app)

# Load model
model = load_model("emotion_model.h5")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ✅ THEN DEFINE ROUTES
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Convert image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({
            "emotion": "No face detected",
            "confidence": 0
        })

    # Take first face
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    # Preprocess
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    face = np.reshape(face, (1, 224, 224, 3))

    # Predict
    preds = model.predict(face, verbose=0)
    label = emotion_labels[np.argmax(preds)]
    confidence = float(np.max(preds))

    return jsonify({
        "emotion": label,
        "confidence": confidence
    })


# ✅ RUN APP LAST
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)