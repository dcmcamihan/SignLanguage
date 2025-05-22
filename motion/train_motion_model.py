import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'motion_data.csv')
model_path = os.path.join(script_dir, 'asl_motion_model.h5')
labels_path = os.path.join(script_dir, 'motion_labels.txt')

df = pd.read_csv(csv_path)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(set(y_encoded))

# Save labels
with open(labels_path, 'w') as f:
    for label in encoder.classes_:
        f.write(label + '\n')

# Reshape to (samples, sequence, features)
SEQ_LENGTH = 30
X = X.reshape((-1, SEQ_LENGTH, 63))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 63)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))
model.save(model_path)
print(f"Motion model saved to {model_path}")
