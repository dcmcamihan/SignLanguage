import cv2
import mediapipe as mp
import pandas as pd
import os

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set path to CSV file in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'hand_sign_data_tf.csv')

# Create CSV if it doesn't exist or fix malformed headers if necessary
if not os.path.exists(csv_path):
    # Create a new CSV with the correct header structure
    pd.DataFrame(columns=["label"] + [f"x{i}" for i in range(21)] +
                 [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]).to_csv(csv_path, index=False)
else:
    try:
        # Try reading the CSV and checking for column consistency
        df = pd.read_csv(csv_path)
        expected_columns = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]

        # If the number of columns doesn't match the expected, fix the header
        if df.shape[1] != len(expected_columns):
            print("Fixing malformed CSV header...")
            df = pd.read_csv(csv_path, header=None, skiprows=1)
            df.columns = expected_columns
            df.to_csv(csv_path, index=False)
            print("CSV header fixed.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # If CSV is malformed, recreate the file
        pd.DataFrame(columns=["label"] + [f"x{i}" for i in range(21)] +
                     [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]).to_csv(csv_path, index=False)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press a key (e.g., 'a', 'b') to save current hand landmarks.")
print("Press ESC to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip image and convert to RGB
    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Display instruction on screen
    cv2.putText(image, "Press key to label hand pose, ESC to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    cv2.imshow('Hand Sign Data Collector', image)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break
    elif landmarks and key != 255:
        label = chr(key)
        data = [label] + landmarks
        df = pd.DataFrame([data])
        df.to_csv(csv_path, mode='a', header=not os.path.getsize(csv_path), index=False)
        print(f"Saved data for label: '{label}'")

cap.release()
cv2.destroyAllWindows()