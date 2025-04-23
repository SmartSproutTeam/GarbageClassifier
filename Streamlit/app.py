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

st.title("FridGPT")
st.write("Upload an image for prediction")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # Preprocess the image if needed (e.g., resize, grayscale, flatten)
    # Example for flattening image:
    img_flattened = img_array.flatten().reshape(1, -1)

    # Make prediction
    prediction = model.predict(img_flattened)
    probability = getattr(model, "predict_proba", lambda x: None)(img_flattened)

    st.subheader("Prediction")
    st.write(prediction[0])

    if probability is not None:
        st.subheader("Probabilities")
        st.write(probability[0])
