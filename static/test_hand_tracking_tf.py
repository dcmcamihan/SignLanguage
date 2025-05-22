import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'asl_tf_model.h5')
labels_path = os.path.join(script_dir, 'labels.txt')

# Load model and labels
model = tf.keras.models.load_model(model_path)
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Running real-time sign prediction... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark data
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict(np.array([landmarks]))[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]

            # Display label
            cv2.putText(image, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Prediction", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()