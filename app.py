from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('digit_recognition_model.h5')  # Load trained model

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

def preprocess_image(image_path):
    """Preprocess uploaded image for prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Preprocess and predict
            image = preprocess_image(filepath)
            prediction = model.predict(image)
            predicted_digit = np.argmax(prediction)

            return render_template('index.html', filename=file.filename, prediction=predicted_digit)

    return render_template('index.html', filename=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)