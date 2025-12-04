# app.py
import streamlit as st
import numpy as np
import cv2
from predict import predict_disease
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import base64
import time

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="🌿 AgroBot — Plant Disease Detector",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------
# Disease descriptions (assistant-written)
# -------------------------
DISEASE_INFO = {
    "fallen_leaf": {
        "title": "Fallen / Drying Leaf",
        "desc": "Signs of drying or tissue senescence — often caused by water stress, physical damage, or late-stage nutrient deficiency. Leaves may curl, brown, or fall off. Early diagnosis helps targeted nutrient or irrigation correction.",
        "advice": "Check soil moisture and root health. Provide foliar nutrient spray if deficiency suspected. Avoid overwatering."
    },
    "powdery": {
        "title": "Powdery Mildew",
        "desc": "White to gray powdery fungal layer on the upper surface of leaves. Powdery mildew reduces photosynthesis and weakens plants if untreated.",
        "advice": "Remove heavily infected leaves, improve airflow, and apply a recommended fungicide or neem-based treatment."
    },
    "rust": {
        "title": "Rust Disease",
        "desc": "Small yellow, orange or brown pustules (rust-like spots) on leaf surfaces caused by rust fungi. This can spread rapidly in humid conditions.",
        "advice": "Isolate infected plants, remove infected tissue and apply fungicidal sprays as per guidelines."
    },
    "healthy": {
        "title": "Healthy Leaf",
        "desc": "No visible disease symptoms found. Leaf shows normal color, texture and turgor.",
        "advice": "Continue regular monitoring and good cultural practices (balanced nutrition, irrigation, pest scouting)."
    }
}

# -------------------------
# Custom CSS
# -------------------------
st.markdown(
    """
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(120deg, #e6f7ff 0%, #f7fff5 25%, #fff7f0 50%, #f5f1ff 75%);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Glass card */
    .glass {
        background: rgba(255,255,255,0.65);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(31, 38, 135, 0.12);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }

    /* Header */
    .main-title {
        font-size:40px;
        font-weight:800;
        text-align:center;
        margin-bottom: 6px;
        color: #0b6b3a;
        letter-spacing: 0.6px;
    }
    .sub-title {
        text-align:center;
        color:#2d2d2d;
        margin-top: -6px;
        margin-bottom: 18px;
    }

    /* Upload area */
    .upload-box {
        border: 2px dashed rgba(15,99,67,0.12);
        border-radius: 12px;
        padding: 12px;
        text-align:center;
    }

    /* Result card */
    .result {
        padding: 12px;
        border-radius: 12px;
        text-align:center;
    }

    .footer {
        text-align:center;
        color:#666;
        font-size:12px;
        margin-top:18px;
    }

    /* small responsive tweaks */
    @media (max-width: 600px) {
        .main-title { font-size:28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["Upload Image", "Camera Snapshot"], index=0)
    show_info = st.checkbox("Show disease info after prediction", value=True)
    pdf_quality = st.select_slider("PDF Image Quality", ["Low", "Medium", "High"], value="Medium")
    st.markdown("---")
    st.caption("AgroBot — Plant disease detection UI\nMade with ❤️")

# -------------------------
# Header
# -------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>🌿 AgroBot — Plant Disease Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload or capture a leaf image to detect disease. Fast, offline, and simple.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("#### Drag & drop an image here or use the camera")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### Quick Actions")
    st.write("• Model: Keras `.h5` (agrobot.h5)")
    st.write("• Classes: fallen_leaf, powdery, rust, healthy")
    st.write("• Confidence & PDF report included")
    st.button("💡 Tips", help="Place camera ~20-30 cm above leaf for consistent detection")

st.markdown("</div>", unsafe_allow_html=True)  # close header glass

# -------------------------
# Helper: convert numpy image to bytes for PDF
# -------------------------
def pil_image_to_bytes(pil_img, quality="Medium"):
    bio = BytesIO()
    q = 60 if quality == "Low" else (80 if quality == "Medium" else 95)
    pil_img.save(bio, format="JPEG", quality=q)
    bio.seek(0)
    return bio.read()

# -------------------------
# Generate PDF
# -------------------------
def generate_pdf(pil_img: Image.Image, disease: str, confidence: float, info: dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    # PDF TITLE (ASCII only)
    pdf.cell(0, 10, "AgroBot Plant Disease Report", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", size=12)

    # ALL ASCII — remove emojis
    pdf.cell(0, 8, f"Disease: {info['title']}", ln=True)
    pdf.cell(0, 8, f"Predicted Label: {disease}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence*100:.2f}%", ln=True)
    pdf.ln(4)

    pdf.multi_cell(0, 6, f"Description: {info['desc']}")
    pdf.ln(3)
    pdf.multi_cell(0, 6, f"Suggested Action: {info['advice']}")
    pdf.ln(6)

    # Insert leaf image
    # Convert to JPG bytes
    bio = BytesIO()
    pil_img.save(bio, format="JPEG", quality=85)
    bio.seek(0)

    pdf.image(bio, x=15, w=180)

    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out


# -------------------------
# Main input & prediction flow
# -------------------------
image_for_display = None
prediction_result = None

# Camera snapshot mode
if mode == "Camera Snapshot":
    st.markdown("### 📷 Capture from camera")
    cam_img = st.camera_input("Use your webcam (single snapshot).")
    if cam_img is not None:
        img_bytes = cam_img.getvalue()
        np_img = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        image_for_display = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# Upload mode
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_for_display = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# Show image preview & run prediction
if image_for_display is not None:
    st.markdown("<div class='glass' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.image(image_for_display, caption="📸 Input Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button("🔍 Predict Disease"):
        with st.spinner("Running model..."):
            # convert to BGR for predict.py which uses cv2 resize etc.
            img_bgr = cv2.cvtColor(image_for_display, cv2.COLOR_RGB2BGR)
            label, conf = predict_disease(img_bgr)
            prediction_result = (label, conf)

            # show animated success
            time.sleep(0.4)

    # If prediction available show results
if prediction_result:
    label, conf = prediction_result
    info = DISEASE_INFO.get(label, {
        "title": label,
        "desc": "No description available.",
        "advice": ""
    })

    # styled result card
    st.markdown("<div class='glass' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("### 🔎 Prediction Result")
    # colored header based on label
    color_map = {
        "healthy": "#2ecc71",
        "fallen_leaf": "#e67e22",
        "powdery": "#8e44ad",
        "rust": "#c0392b"
    }
    header_color = color_map.get(label, "#333333")
    st.markdown(f"<h2 style='color:{header_color}; margin-top:6px;'>{info['title']}</h2>", unsafe_allow_html=True)

    # confidence bar
    conf_pct = int(conf * 100)
    st.progress(conf_pct)

    st.write(f"**Label:** `{label}`")
    st.write(f"**Confidence:** {conf*100:.2f}%")

    # Short advice box
    if label == "healthy":
        st.success("🌱 This leaf appears healthy.")
    else:
        st.error(f"⚠️ {info['title']} detected — recommended action below.")

    # Modal / Expandable disease info
    if show_info:
        with st.expander("🛈 Disease Details & Advice", expanded=True):
            st.write("**Description:**")
            st.write(info["desc"])
            st.write("**Suggested Action:**")
            st.write(info["advice"])

    # PDF generation
    pil_img = Image.fromarray(image_for_display)
    pdf_file = generate_pdf(pil_img, label, conf, info)
    b64 = base64.b64encode(pdf_file.read()).decode()
    pdf_file.seek(0)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_file,
        file_name="agrobot_report.pdf",
        mime="application/pdf"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>Made with ❤️ • AgroBot Plant Disease Detector • Tip: Use consistent lighting & camera height (20–30 cm)</div>", unsafe_allow_html=True)
