import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PneumoScan - Pneumonia Detection",
    page_icon="🩺",
    layout="centered",
)

# -----------------------------
# Loading the trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Load your trained weights
    model.load_state_dict(torch.load("model/pneumonia_resnet18_fast.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# UI Styling Compatible with Light and Dark mode)
# -----------------------------
st.markdown("""
<style>
    /* Center and style title */
    .title {
        font-size: 42px !important;
        font-weight: 700;
        text-align: center;
        margin-top: -30px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    /* Result box adaptive to dark/light */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    /* Auto-detect dark mode */
    @media (prefers-color-scheme: dark) {
        .title { color: #ecf0f1 !important; }
        .subtitle { color: #bdc3c7 !important; }
    }

    @media (prefers-color-scheme: light) {
        .title { color: #2d3436 !important; }
        .subtitle { color: #636e72 !important; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<p class="title">🩺 PneumoScan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect pneumonia from chest X-ray images using a deep learning model.</p>',
            unsafe_allow_html=True)

st.write("")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG or PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()

    class_names = ["Normal", "Pneumonia"]
    prediction = class_names[pred_class]
    confidence = probabilities[pred_class].item() * 100

    # Color-coded result
    if prediction == "Normal":
        color = "#27ae60"   # green
    else:
        color = "#c0392b"   # red

    st.markdown(
        f'<div class="result-box" style="background-color:{color};">'
        f'Prediction: {prediction}<br>Confidence: {confidence:.2f}%'
        f'</div>',
        unsafe_allow_html=True
    )

else:
    st.info("Please upload a chest X-ray image to begin.")
