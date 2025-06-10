import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('/Users/diannaclairemarieamihan/Workspace/Architecture and Organization/SeGlo/TensorFlow/static/asl_tf_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open('asl_tf_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as asl_tf_model.tflite")