import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image.convert('L'))  # Convert to grayscale
    image = cv2.resize(image, (28, 28))   # Resize to 28x28
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Streamlit UI
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image, and the model will predict the number!")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    st.write(f"### 🧠 Predicted Digit: {predicted_digit}")