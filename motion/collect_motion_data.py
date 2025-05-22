import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from collections import deque

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Config
SEQ_LENGTH = 30  # number of frames per gesture
landmark_dim = 21 * 3
buffer = deque(maxlen=SEQ_LENGTH)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'motion_data.csv')

# Create CSV if not exists
if not os.path.exists(csv_path):
    cols = ['label'] + [f"{coord}{i}_{f}" for f in range(SEQ_LENGTH) for coord in ['x', 'y', 'z'] for i in range(21)]
    pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

# Start webcam
cap = cv2.VideoCapture(0)
print("Perform the motion gesture, then press any key to save it. Press ESC to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])

            buffer.append(frame_landmarks)

    # Display instructions
    cv2.putText(image, "Perform motion, press key to save | ESC to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Motion Data Collector', image)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif len(buffer) == SEQ_LENGTH and key != 255:
        print("Enter label for this gesture (e.g., 'hello', 'thank you'):")
        label = input("> ").strip().lower()
        if label:
            flattened = np.array(buffer).flatten()
            df = pd.DataFrame([[label] + list(flattened)])
            df.to_csv(csv_path, mode='a', header=not os.path.getsize(csv_path), index=False)
            print(f"Saved sequence for label: '{label}'")
            buffer.clear()  # reset after saving

cap.release()
cv2.destroyAllWindows()
