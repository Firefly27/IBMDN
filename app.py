from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, redirect, render_template, request
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img,img_to_array
from keras.models import load_model
import json

import secrets
from flask import Flask, flash, render_template, request, redirect, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)

global graph
graph=tf.compat.v1.get_default_graph()


predictions=[
                'Bear Mammal',
 'Bluebell Flower',
 'ColtsFoot Flower',
 'Corpse Flower',
 'Cow Mammal',
 'Daisy Flower',
 'Dandelion Flower',
 'Duck Bird',
 'Eagle Bird',
 'Elephant Mammal',
 'Flamingo Bird',
 'Fox Mammal',
 'Great Indian Bustard Bird',
 'Hornbill Bird',
 'Horse Mammal',
 'Hummingbird Bird',
 'Lady Slipper Orchid Flower',
 'Leopard Mammal',
 'Owl Bird',
 'Panda Mammal',
 'Pangolin Mammal',
 'Parrot Bird',
 'Pigeon Bird',
 'Rat Mammal',
 'Rose Flower',
 'Senenca White Deer Mammal',
 'Spoon Billed Sandpiper Bird',
 'Sunflower Flower',
 'Tulip Flower',
 'Windflower Flower'
            ]

app = Flask(__name__)
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = ROOT_DIR+"/static/uploads/"
app.config["SECRET_KEY"] = str(secrets.SystemRandom().getrandbits(128))
configure_uploads(app, photos)



@app.route('/', methods=['GET'])

def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])

def upload():

    if request.method=='GET':
        return render_template('upload.html')


    if request.method == 'POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'static/uploads',secure_filename(f.filename))
        image_file=os.path.join("static/uploads",secure_filename(f.filename))
        if os.path.isdir(file_path):
             return render_template('upload.html',error="Please upload an image file")
        f.save(file_path)
        img=load_img(file_path,target_size=(229,229))
        x=img_to_array(img)
        x = preprocess_input(x)
        inp = np.array([x])

        with graph.as_default():
            loaded_model=load_model("final_model/final_model.h5")
            preds =  np.argmax(loaded_model.predict(inp),axis=1)

        text = predictions[preds[0]]

        with open('predictions.json', 'r') as f:
            data = json.load(f)
            description =data[str(preds[0])].get('description')
            species =predictions[preds[0]]
            species_type =data[str(preds[0])].get('type')

        return render_template('upload.html',uploaded_image=image_file,description=description,species=species,type=species_type)



if __name__=='__main__':
    app.run(debug=False)

