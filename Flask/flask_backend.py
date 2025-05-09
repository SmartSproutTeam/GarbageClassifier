from flask import Flask, request, jsonify
import pickle
from PIL import Image
import io
import numpy as np

app = Flask(__name__)


def load_model():
    model_path = "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
label_namges = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

# Load the YOLOv5 model
# model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # uses local clone

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    img = image.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize if model expects that
    img_reshaped = img_array.reshape(1, 128, 128, 3)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_reshaped)
    pred_index = np.argmax(prediction)
    predicted_label = label_namges[pred_index]
    confidence = prediction[0][pred_index]
    
    return predicted_label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)

