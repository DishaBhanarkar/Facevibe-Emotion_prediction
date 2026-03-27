import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

# Emotion labels (IMPORTANT: same order as training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        # Predict
        preds = model.predict(face, verbose=0)

        confidence = np.max(preds)
        label_index = np.argmax(preds)

        emotion = emotion_labels[label_index]
        emotion_text = f"{emotion} ({confidence:.2f})"
        
        # Set color based on emotion
        if emotion == "happy":
            color = (0, 255, 0)        # Green
        elif emotion == "sad":
            color = (255, 0, 0)        # Blue
        elif emotion == "angry":
            color = (0, 0, 255)        # Red
        elif emotion == "surprise":
            color = (0, 255, 255)      # Yellow
        elif emotion == "fear":
            color = (255, 0, 255)      # Purple
        elif emotion == "disgust":
            color = (0, 128, 0)        # Dark Green
        else:
            color = (200, 200, 200)    # Neutral (Gray)

        # Draw box + label
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, emotion_text, (x+5, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()