import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# --- Page title ---
st.title("Brain Tumor Image Segmentation with TensorFlow")

# --- Hugging Face Model Link ---
MODEL_URL = "https://huggingface.co/yashika2212/brain-tumor/resolve/main/model2.h5"
MODEL_PATH = "model2.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Hugging Face..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

def load_segmentation_model(model_path):
    """Load the segmentation model from the given path."""
    try:
        model = load_model(model_path, compile=False)  # Avoid compilation issues
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess the input image for model prediction."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_mask(model, image):
    """Generate the predicted mask using the segmentation model."""
    pred_mask = model.predict(image, verbose=0)
    pred_mask = np.squeeze(pred_mask)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    return pred_mask

def overlay_mask_on_image(image, mask, mask_color=(0, 255, 150), alpha=0.5):
    """Overlay the predicted mask on the input image with transparency."""
    image = np.squeeze(image)  # Remove batch dimension
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_colored = np.zeros_like(image)
    mask_colored[mask == 1] = mask_color
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return blended

# --- Upload image ---
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# --- Predict button ---
if st.button("Predict"):
    if image_file is not None:
        # Read image
        image = np.array(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Load model
        model = load_segmentation_model(MODEL_PATH)

        if model:
            preprocessed_image = preprocess_image(image)
            pred_mask = predict_mask(model, preprocessed_image)
            overlay = overlay_mask_on_image(image, pred_mask)
            st.image(overlay, caption="Output Image with Mask", use_column_width=True)
    else:
        st.warning("Please upload an image to make predictions.")

        st.error("Please upload an image.")
