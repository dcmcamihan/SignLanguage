import tensorflow as tf
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'asl_motion_model.h5')
tflite_model_path = os.path.join(script_dir, 'asl_motion_model.tflite')

# Load the motion model
model = tf.keras.models.load_model(model_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the .tflite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite motion model saved as {tflite_model_path}")