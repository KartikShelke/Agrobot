import streamlit as st
import numpy as np
import cv2
from predict import predict_disease

# ===========================
# 🎨 Custom CSS Styling
# ===========================
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #0f9b0f, #00b09b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #444;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# 🌿 App Header
# ===========================
st.markdown("<h1 class='title'>🌿 Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a leaf image for instant disease prediction using AgroBot Model</p>",
            unsafe_allow_html=True)

# ===========================
# 📤 Upload Image
# ===========================
uploaded = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="🌱 Uploaded Leaf", use_column_width=True)

    # ===========================
    # 🔍 Prediction
    # ===========================
    disease, confidence = predict_disease(image)

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    st.markdown(f"<h2 style='color:#008037;'>🔍 Prediction Result</h2>", unsafe_allow_html=True)

    # ---------------------------
    # Disease Color Coding
    # ---------------------------
    colors = {
        "healthy": "#2ecc71",
        "fallen_leaf": "#e67e22",
        "powdery": "#9b59b6",
        "rust": "#c0392b"
    }

    color = colors.get(disease, "#333")

    st.markdown(
        f"<h3 style='color:{color}; font-weight:700;'>🩺 Disease: {disease.replace('_',' ').title()}</h3>",
        unsafe_allow_html=True
    )

    st.progress(confidence)

    st.write(f"**Confidence:** {confidence*100:.2f}%")

    if disease == "healthy":
        st.success("🌱 This leaf looks healthy!")
    else:
        st.error(f"⚠️ Disease detected: {disease}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# 📝 Footer
# ===========================
st.markdown(
    "<p class='footer'>Developed using your AgroBot Plant Disease Model • Streamlit UI by ChatGPT</p>",
    unsafe_allow_html=True
)
