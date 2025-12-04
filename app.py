# app.py
import streamlit as st
import numpy as np
import cv2
from predict import predict_disease

st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
st.title("🌿 Plant Disease Detection (AgroBot Model)")
st.write("Upload a leaf image to detect disease using your trained model.")

uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Prediction
    disease, confidence = predict_disease(image)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Disease:** {disease}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Styling
    if disease == "healthy":
        st.success("🌱 The leaf looks healthy!")
    else:
        st.error(f"⚠️ Disease Detected: {disease}")
