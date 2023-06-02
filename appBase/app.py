# system library
# import os

# math and tables
# import pandas as pd
import numpy as np

# for model building
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
tf.random.set_seed(42)

# visualization libraries
import cv2
# import matplotlib.pyplot as plt

# some utils
# from sklearn.model_selection import train_test_split
# from random import randint

from tensorflow.keras.models import load_model
from multiprocessing import Pool
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import cv2
import base64
import numpy as np









# data_path = 'Dog Emotion\\'

# class_names = sorted(os.listdir(data_path))
# # remove labels.csv because it's not a class
# class_names.remove('labels.csv')
# num_classes = len(class_names)

img_size = (192, 192, 3)

# print(f'{num_classes} classes: {class_names}\nimage size: {img_size}')

# images = []
# labels = []
# labels_df = pd.read_csv('Dog Emotion\labels.csv')

# print('\n\nlabels dataframe: \n', labels_df.head())

# for image in labels_df.iloc:
#     images.append(np.asarray(cv2.resize(cv2.imread(data_path + image[2] + '/' + image[1], cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1]))
    
#     # labels will be in the form of a vector: [0, 1, 0, 0] or [1, 0, 0, 0]
#     label = np.zeros(num_classes)
#     label[class_names.index(image[2])] = 1
#     labels.append(label)

# labels = np.asarray(labels)
# images = np.asarray(images)

# print(f'\nlabels shape: {labels.shape}')
# print(f'images shape: {images.shape}')















# Display 16 pictures from the dataset
# fig, axs = plt.subplots(4, 4, figsize=(11, 11))

# for x in range(4):
#     for y in range(4):
#         i = randint(0, len(images))
        
#         axs[x][y].imshow(images[i])
        
#         # delete x and y ticks and set x label as picture label
#         axs[x][y].set_xticks([])
#         axs[x][y].set_yticks([])
#         axs[x][y].set_xlabel(class_names[np.argmax(labels[i])])
        
# plt.show()




# X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.15, random_state=42)

# print(f'train images shape: {X_train.shape}\ntrain labels shape: {y_train.shape}\n\nvalidation images shape: {X_val.shape}\nvalidation labels shape: {y_val.shape}\n')




# cnn_model = tf.keras.Sequential()

# # Inputs and rescaling
# cnn_model.add(tf.keras.layers.Rescaling(scale=1. / 255, input_shape=(img_size)))

# # Convolutional block 1
# cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# # Convolutional block 2
# cnn_model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# # Convolutional block 3
# cnn_model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# # Convolutional block 4
# cnn_model.add(tf.keras.layers.Conv2D(512, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.Conv2D(512, (2, 2), activation='relu', padding='same'))
# cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# cnn_model.add(tf.keras.layers.Flatten())

# # Dense block
# cnn_model.add(tf.keras.layers.Dense(256, activation='relu'))
# cnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
# cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
# cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


# cnn_model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])

# cnn_model.summary()



# # creating ModelChecpoint callback
# checkpoint_callback = ModelCheckpoint('cnn_model/model{epoch:02d}')

# history = cnn_model.fit(images, labels, epochs=25, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])



# accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(accuracy))

# plt.figure()
# plt.plot(epochs, accuracy, label='Training Accuracy')
# plt.plot(epochs, loss, label='Training Loss')
# plt.legend()
# plt.title('Training Accuracy and Loss')

# plt.figure()
# plt.plot(epochs, val_accuracy, label='Validation Accuracy')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.legend()
# plt.title('Validation Accuracy and Loss')

# plt.show()

def classifyImage(dog_test_image):
    prediction = cnn_model.predict(dog_test_image[None, ...])[0]
    return class_names[np.argmax(prediction)]


def classifyImages(img_list):
    with Pool(4) as p:
        results = p.map(classifyImage, img_list)
    return results


def classifyImagesSequential(img_list):
    prediction = classifyImage(img_list[0])
    prediction2 = classifyImage(img_list[1])
    prediction3 = classifyImage(img_list[2])
    prediction4 = classifyImage(img_list[3])
    return [prediction, prediction2, prediction3, prediction4]

class_names = ['angry', 'happy', 'relaxed', 'sad']
cnn_model = load_model('cnn_model/model25')


if __name__ == '__main__':
    # freeze_support()
    print("Executando!")

    data_path = 'Dog Emotion\\'
    test = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    test2 = cv2.imread('test2.jpg', cv2.IMREAD_COLOR)
    test3 = cv2.imread('test3.jpg', cv2.IMREAD_COLOR)
    test4 = cv2.imread('test4.jpg', cv2.IMREAD_COLOR)
    # print(test)
    # print(test2)
    # print(test3)
    # print(test4)

    test = np.asarray(cv2.resize(test, img_size[0:2])[:, :, ::-1])
    test2 = np.asarray(cv2.resize(test2, img_size[0:2])[:, :, ::-1])
    test3 = np.asarray(cv2.resize(test3, img_size[0:2])[:, :, ::-1])
    test4 = np.asarray(cv2.resize(test4, img_size[0:2])[:, :, ::-1])

    img_list = [test, test2, test3, test4]
    # print(img_list)

    start = datetime.now()
    predictions = classifyImagesSequential(img_list)
    end = datetime.now()
    print(predictions)
    print(end - start)

    start = datetime.now()
    predictions = classifyImages(img_list)
    end = datetime.now()
    print(predictions)
    print(end - start)






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
        foto = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Process the image
        # cv2.imshow("Imagem",foto)
        # cv2.waitKey(0)
    return "A"

app.run(port=5000, host='localhost', debug=True)




########################################

# cnn_model = load_model('cnn_model/model25')

# dog_test_path = "test.jpg"

# dog_test_image = np.asarray(cv2.resize(cv2.imread(dog_test_path, cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1])

# prediction = cnn_model.predict(dog_test_image[None, ...])[0]

# print('############################\n')
# print(f'IMAGE PREDICTION: {class_names[np.argmax(prediction)]}')
# print('\n############################')

########################################















# fig, axs = plt.subplots(7, 4, figsize=(15, 15))

# i = 0
# for x in range(7):
#     for y in range(4):
#         prediction = cnn_model.predict(X_val[i][None, ...], verbose=0)[0]
        
#         axs[x][y].set_xticks([])
#         axs[x][y].set_yticks([])
#         axs[x][y].set_xlabel(f'prediction: {class_names[np.argmax(prediction)]} | label: {class_names[np.argmax(y_val[i])]}')
        
#         axs[x][y].imshow(X_val[i])
        
#         i += 1
# plt.show()

# print('Finished!')