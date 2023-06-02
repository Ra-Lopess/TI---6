from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import cv2
import base64
import numpy as np
import sys

def image_to_base64():
    image_path = 'C:/Users/USER/Desktop/clei.jpg'
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_data = base64.b64encode(image_data)
        base64_string = base64_data.decode('utf-8')
        return base64_string

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

    for img in imgs:
        name = img['name']
        
        base64_data = img['data']
        print(name)

        image_data = base64.b64decode(base64_data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        print("=================================\n")
        print(np_array)
        foto = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Process the image
        cv2.imshow("Imagem",foto)
        cv2.waitKey(0)
    return "A"

app.run(port=5000, host='localhost', debug=True)

# Example usage

