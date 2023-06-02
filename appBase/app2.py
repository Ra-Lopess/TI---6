#   base Imports
import base64

#   flask Imports
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

#  cv2 Imports
import cv2
import numpy as np

#   CNN imports
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

#   paralelization imports
from multiprocessing.pool import ThreadPool

# OpenCV Param
img_size = (192, 192, 3)

# CNN Param
tf.random.set_seed(42)
class_names = ['angry', 'happy', 'relaxed', 'sad']
cnn_model = load_model('cnn_model/model25')


def classifyImage(dog_test_image):
    prediction = cnn_model.predict(dog_test_image[None, ...])[0]
    return class_names[np.argmax(prediction)]


def classifyImages(img_list):
    with ThreadPool(6) as p:
        results = p.map(classifyImage, img_list)
    return results




# Flask Application
app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/')
def index():
    print("API no Ar!")
    return "API no Ar!"

@app.route('/cnn', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_image():
    json_data = request.get_json()
    imgs = json_data['images']
    img_list = []
    for img in imgs:
        name = img['name']
        base64_data = img['data']

        image_data = base64.b64decode(base64_data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        foto = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        new_np_array = np.asarray(cv2.resize(foto, img_size[0:2])[:, :, ::-1])
        # prediction = cnn_model.predict(new_np_array[None, ...])[0]
        img_list.append(new_np_array)
        
    predictions = classifyImages(img_list)
    # emotions.append(class_names[np.argmax(prediction)])
    print(predictions)
    return predictions

app.run(port=5000, host='localhost', debug=True)