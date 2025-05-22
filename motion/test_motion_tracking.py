import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
script_dir = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(script_dir, 'asl_motion_model.h5'))

with open(os.path.join(script_dir, 'motion_labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

SEQ_LENGTH = 30
buffer = deque(maxlen=SEQ_LENGTH)

cap = cv2.VideoCapture(0)
print("Running real-time motion prediction... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            buffer.append(landmarks)

    if len(buffer) == SEQ_LENGTH:
        input_data = np.expand_dims(np.array(buffer), axis=0)
        prediction = model.predict(input_data)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        confidence = prediction[predicted_index]

        cv2.putText(image, f"{predicted_label} ({confidence:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Motion Sign Prediction", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
