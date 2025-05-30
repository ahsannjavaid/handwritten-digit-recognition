from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import eventlet
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Load trained model
# model_name = 'digit_recognition_model.h5'
model_name = 'cnn_digit_recognition.h5'
model = tf.keras.models.load_model(model_name)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    """Preprocess the image for prediction."""
    if image.mode == 'RGBA':
        # Giving canvas image the white BG since it is transparent
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        
        # Paste the original image on the white background
        white_bg.paste(image, (0, 0), image)
        image = white_bg
        
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = Image.open(filepath)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)

            return render_template('index.html', filename=file.filename, prediction=predicted_digit)

    return render_template('index.html', filename=None, prediction=None)

@socketio.on('canvas_data')
def handle_canvas(data):
    """Handle canvas image data from frontend."""
    image_data = data.replace("data:image/png;base64,", "")
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Save drawn image
    drawn_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "drawn_digit.png")
    image.save(drawn_image_path)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = int(np.argmax(prediction))

    socketio.emit('prediction_result', {'prediction': predicted_digit})

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded or drawn images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)