import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

model = load_model("plant_disease_cnn.h5")

classes = ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Healthy Apple']  # Example; replace based on your dataset

st.title("Plant Disease Classifier")
st.write("Upload a leaf image and the model will predict the disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]

    st.success(f"Prediction: **{predicted_class}**")
