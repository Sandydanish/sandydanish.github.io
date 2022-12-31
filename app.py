from flask import Flask, flash, redirect, url_for, request, render_template, session, Response
import urllib.request
import base64
import os
from werkzeug.utils import secure_filename
from io import BytesIO

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import cv2
import colorsys
import argparse
import imutils
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import imageio
# import imgaug as ia
# import imgaug.augmenters as iaa

import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.augmentations.transforms import ChannelShuffle

import cv2
from db import db_init, db
from models import Img
import uuid as uuid

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

LABELS = ['Akita', 'Alaskan Malamute', 'Basset', 'Beagle', 'Border Collie', 'Bulldog', 'Chihuahua',
          'Chow', 'Dalmatian', 'Golden Retriever', 'Maltese', 'Pekingese', 'Pomeranian', 'Poodle',
          'Shiba Inu', 'Shih Tzu', 'Siberian Husky']

# CNN
encoder = LabelBinarizer()

y = encoder.fit_transform(np.array(LABELS))

modelCNN = Sequential()

modelCNN.add(Conv2D(64,(3,3),activation="relu", padding="same"))
modelCNN.add(MaxPooling2D())
modelCNN.add(Dropout(0.2))

modelCNN.add(Conv2D(64,(3,3),activation="relu", padding="same"))
modelCNN.add(MaxPooling2D())
modelCNN.add(Dropout(0.2))

modelCNN.add(Conv2D(64,(3,3),activation="relu", padding="same"))
modelCNN.add(MaxPooling2D())
modelCNN.add(Dropout(0.2))

modelCNN.add(Conv2D(128,(3,3),activation="relu", padding="same"))
modelCNN.add(MaxPooling2D())

modelCNN.add(Conv2D(128,(3,3),activation="relu", padding="same"))
modelCNN.add(MaxPooling2D())

modelCNN.add(Flatten())

modelCNN.add(Dense(1024,activation="relu"))
modelCNN.add(Dropout(0.5))

modelCNN.add(Dense(256,activation="relu"))
modelCNN.add(Dropout(0.2))

modelCNN.add(Dense(len(LABELS),activation="softmax"))

base_model=VGG16(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(512,activation='relu')(x)
preds=Dense(len(LABELS),activation='softmax')(x)

modelCNN=Model(inputs=base_model.input,outputs=preds)

for layer in modelCNN.layers[:-5]:
    layer.trainable=False
for layer in modelCNN.layers[-5:]:
    layer.trainable=True
    
modelCNN.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])

print(modelCNN.summary())

modelCNN.load_weights("dogClassificationModelEpoch30.h5")

# MASK RCNN
class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # number of classes on COCO dataset
    NUM_CLASSES = 81

config = SimpleConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
model.load_weights("mask_rcnn_coco.h5", by_name=True)

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = "secret_key"
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db_init(app)
# app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

def allowed_file(filename):
   return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        #Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        #Set UUID
        # img_name = str(uuid.uudi1()) + "_" + img_filename
        #Saving image
        # uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER']), img_name)
        # print(img_filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        # img = Img(img=uploaded_img.read(), mimetype=mimetype, name="imageUpload")
        # db.session.add(img)
        # db.session.commit()

        return render_template('imageUpload2.html')

@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    # print(img_file_path)
    # print(type(img_file_path))
    # img_file_path = Img.query.filter_by(name="imageUpload").first()
    # image_path = base64.b64encode(img_file_path.img).decode('utf-8')
    # print(image_path)
    # print(type(image_path))
    image = cv2.imread(img_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imutils.resize(image, width=512)

    result = model.detect([image], verbose=1)

    r1 = result[0]
    # visualize.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'], class_names, r1['scores'])

    dogclass = np.where(r1['class_ids'] == ([17]))
    if((r1['class_ids'] == [17]).any()):
            dogclassList = dogclass[0].tolist()
            if(len(dogclassList) > 1):
                while(len(dogclassList)>1):
                    dogclassList.pop(1)
                    dogclass2 = dogclassList[0]
                    print(dogclass2)
                    x = r1['rois'][dogclass2][0]
                    y = r1['rois'][dogclass2][1]
                    width = r1['rois'][dogclass2][2]
                    height = r1['rois'][dogclass2][3]
                    print(x, y, width, height)
                    crop_img = image[x:width, y:height]
                    crop_img = cv2.resize(crop_img, (224,224))
                    crop_img = crop_img.reshape(1, 224, 224, 3)
                    predictions = modelCNN.predict(crop_img)
                    label_predictions = encoder.inverse_transform(predictions)
                    im = Image.fromarray(crop_img.squeeze())
                    im.save("staticFiles/uploads/croppedImage.jpg")
                    dogImage = "croppedImage.jpg"
                    im.save(os.path.join(app.config['UPLOAD_FOLDER'], dogImage))
                    session['dogImage_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], dogImage)
                    dog_file_path = session.get('dogImage_file_path', None)
                    labelpredict = np.array2string(label_predictions)
                    print(labelpredict)
                    images = []
                    # Display image in Flask application web page
                    img_path = labelpredict
                    images.append([dog_file_path, img_path])
                    return render_template('show_image.html', user_image = images)
            else:
                dogclass2 = int(dogclass[0])
                print(dogclass2)
                x = r1['rois'][dogclass2][0]
                y = r1['rois'][dogclass2][1]
                width = r1['rois'][dogclass2][2]
                height = r1['rois'][dogclass2][3]
                print(x, y, width, height)
                crop_img = image[x:width, y:height]
                crop_img = cv2.resize(crop_img, (224,224))
                crop_img = crop_img.reshape(1, 224, 224, 3)
                predictions = modelCNN.predict(crop_img)
                crop_img = crop_img.reshape(224, 224, 3)
                label_predictions = encoder.inverse_transform(predictions)
                im = Image.fromarray(crop_img)
                # resImg = Img(img=im, mimetype='jpeg', name="resultImage")
                # db.session.add(resImg)
                # db.session.commit()
                im.save("staticFiles/uploads/croppedImage.jpg")
                dogImage = "croppedImage.jpg"
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], dogImage))
                session['dogImage_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], dogImage)
                dog_file_path = session.get('dogImage_file_path', None)
                labelpredict = np.array2string(label_predictions)
                # print(labelpredict)
                # print(type(labelpredict))
                images = []
                # Display image in Flask application web page
                # dog_file_path = Img.query.filter_by(name="imageUpload").first()
                # image_path = base64.b64encode(img_file_path.img).decode('utf-8')
                # dog_image = dog_file_path.img
                img_predict = labelpredict
                images.append([dog_file_path,img_predict])
                return render_template('show_image.html', user_image = images)
    else:
        image = cv2.resize(image, (224,224))
        image = image.reshape(1, 224, 224, 3)
        predictions = modelCNN.predict(image)
        label_predictions = encoder.inverse_transform(predictions)
        image = image.reshape(224, 224, 3)
        im = Image.fromarray(image)
        print(im)
        im.save("staticFiles/uploads/croppedImage.jpg")
        dogImage = "croppedImage.jpg"
        im.save(os.path.join(app.config['UPLOAD_FOLDER'], dogImage))
        session['dogImage_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], dogImage)
        dog_file_path = session.get('dogImage_file_path', None)
        labelpredict = np.array2string(label_predictions)
        print(labelpredict)
        images = []
        img_path = labelpredict
        images.append([dog_file_path, img_path])
        return render_template('show_image.html', user_image = images)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == '__main__':
   app.run()