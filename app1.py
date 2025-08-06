import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from utils.model_loader import load_model_and_transform

# --- HOMEPAGE MODIFICATION START ---

# Set a title for your app
st.title("🧠 Brain MRI Image Classification")

# Add some descriptive text
st.write(
    """
    Welcome to the Brain MRI Classification app! This tool uses various deep learning models
    to predict the class of a brain MRI image.
    
    To get started, simply select a model from the dropdown menu and upload an image.
    The app will then use the selected model to analyze the image and provide a prediction.
    """
)

# You can add a sidebar for more information or controls
st.sidebar.header("About this App")
st.sidebar.info(
    "This application demonstrates the power of transfer learning and Streamlit for creating "
    "interactive machine learning web apps."
)

# --- HOMEPAGE MODIFICATION END ---

# 🎯 Model options
model_names = ["VGG16", "ResNet", "EfficientNet", "ViT"]
selected_model = st.selectbox("Select a model", model_names)

# 📂 Load model + transform
model, transform, classes = load_model_and_transform(selected_model)
model.eval()

# 📸 Upload image
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=360)

    # 🔍 Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        if hasattr(output, "logits"):
            probs = torch.nn.functional.softmax(output.logits, dim=1)[0]
        else:
            probs = torch.nn.functional.softmax(output, dim=1)[0]
        
        pred_idx = torch.argmax(probs).item()

    # 🎯 Result
    st.success(f"Predicted: `{classes[pred_idx]}` with {probs[pred_idx]*100:.2f}% confidence")

    # 📊 Bar chart
    st.subheader("Class Probabilities")
    st.bar_chart({classes[i]: float(p) for i, p in enumerate(probs)})