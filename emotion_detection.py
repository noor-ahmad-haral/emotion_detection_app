import streamlit as st
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the FaceNet model
class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 24 * 24, 500)  # Adjust the input size for the fully connected layer
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * 24 * 24)  # Adjust the view size based on the actual size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
num_classes = 7  # Assuming 7 classes for emotions
model = FaceNet(num_classes)
model.load_state_dict(torch.load('emotion_model1.pth'))
model.eval()

# Load the label map
label_map = torch.load('emotion_label_map.pth')

# Define color map for different labels
label_color_map = {
    "angry": "🔥",      # Angry
    "disgust": "🤢",    # Disgust
    "fear": "😨",       # Fear
    "happy": "😄",      # Happy
    "neutral": "😐",    # Neutral
    "sad": "😢",        # Sad
    "surprise": "😲"    # Surprise
}

# Function to preprocess input frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (48, 48))
    frame = frame / 255.0
    frame = torch.Tensor(frame).permute(2, 0, 1).unsqueeze(0)
    return frame

# Function to make predictions on a single frame
def predict_emotion(frame):
    face = preprocess_frame(frame)
    output = model(face)
    _, predicted = torch.max(output, 1)
    label = label_map[int(predicted)]
    return label

def real_time_emotion_recognition():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    st.text("Analyzing emotions...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label = predict_emotion(face)

            # Use the color map to get the emoji for the label
            emoji = label_color_map.get(label, "❓")  # Default emoji if label not found

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, emoji, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        st.image(frame, channels="BGR", use_column_width=True)

        if st.button("Stop 🛑", key=f"stop_button_recognition_{time.time()}"):
           break

    st.success("Emotion analysis stopped.")
    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
st.title("🎭 Real-time Emotion Recognition App 🎥")
st.write("This app uses your webcam for real-time emotion recognition.")

# Streamlit App
if st.button("Start ▶️"):
    real_time_emotion_recognition()
