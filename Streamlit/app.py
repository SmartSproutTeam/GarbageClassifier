import os
import pickle
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Garbage Classifier")
st.write("Upload an image for prediction")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
label_namges = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize if model expects that
    img_reshaped = img_array.reshape(1, 128, 128, 3)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_reshaped)
    pred_index = np.argmax(prediction)
    predicted_label = label_namges[pred_index]
    confidence = prediction[0][pred_index]

    st.subheader("Prediction")
    st.write(f"Predicted label: **{predicted_label}** (Confidence: {confidence:.2f})")