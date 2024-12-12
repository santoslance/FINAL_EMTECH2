import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('weather_model.h5')  # Replace with your model file name
    return model

model = load_model()

st.write("""
# Weather Classification
Upload an image to classify the weather as Cloudy, Rain, Shine, or Sunrise.
""")

# Upload the image
file = st.file_uploader("Choose a weather image from your computer", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (128, 128)  # Adjust to match your model's input size

    # Resize the image
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    # Convert to grayscale or RGB based on your model
    if img.ndim == 2:  # If grayscale, add a channel dimension
        img = img.reshape(img.shape + (1,))
    elif img.shape[2] != 3:  # If not RGB, ensure correct shape
        st.error("The uploaded image must have 3 channels (RGB).")
        return None

    # Normalize the image
    img = img / 255.0  # Ensure pixel values are between 0 and 1

    # Reshape for the model
    img_reshape = img[np.newaxis, ...]

    # Predict using the model
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if prediction is not None:
        class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  # Update with your weather classes
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        result = f"Prediction: {predicted_class} with {confidence:.2f}% confidence"
        st.success(result)
