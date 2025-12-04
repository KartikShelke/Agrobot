# predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load model
model = tf.keras.models.load_model("agrobot_disease_model2.h5")

# Class labels (update according to your model)
CLASS_LABELS = ["fallen_leaf", "healthy", "powdery", "rust"]


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_disease(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return CLASS_LABELS[class_index], float(confidence)
