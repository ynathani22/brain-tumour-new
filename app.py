import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

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

st.title("Image Segmentation with TensorFlow")

model_path = st.text_input("Model Path", "brain.h5")
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if image_file is not None:
        image = np.array(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        model = load_segmentation_model(model_path)
        if model:
            preprocessed_image = preprocess_image(image)
            pred_mask = predict_mask(model, preprocessed_image)
            overlay = overlay_mask_on_image(image, pred_mask)
            st.image(overlay, caption="Output Image with Mask", use_column_width=True)
    else:
        st.error("Please upload an image.")