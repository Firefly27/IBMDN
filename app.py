from __future__ import division, print_function
import os
import numpy as np
import json
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential
from keras.models import model_from_json
from tensorflow.keras.utils import load_img,img_to_array

global graph
graph=tf.compat.v1.get_default_graph()
app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='GET':
        return render_template('upload.html')

    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'static/uploads',secure_filename(f.filename))
        image_file=os.path.join("static/uploads",secure_filename(f.filename))
        f.save(file_path)
        img=load_img(file_path,target_size=(256,256))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        with graph.as_default():
            json_path=r"./final_model/final_model.json"
            json_file=open(json_path,'r')
            loaded_model_json=json_file.read()
            json_file.close()


            h5_path=r"./final_model/final_model.h5"
            loaded_model=model_from_json(loaded_model_json)
            loaded_model.load_weights(h5_path)
            print('Model loaded.Check http://127.0.0.1:5000/')

            preds = np.argmax(loaded_model.predict(x), axis=-1)
            found={
                0: { "Type": "Bird", "Species": "Great Indian Bustard Bird"},
1: {"Type": "Bird", "Species": "Spoon Billed Sandpiper Bird"},
2: {"Type": "Flower", "Species": "Corpse Flower"},
3: {"Type": "Flower", "Species": "Lady Slipper Orchid Flower"},
4: {"Type": "Mammal", "Species": "Pangolin Mammal"},
5: {"Type": "Mammal", "Species": "Senenca White Deer Mammal"},
            }
            text=found[preds[0]]
            return render_template('upload.html',prediction_text=text,uploaded_image=image_file)


if __name__=='__main__':
    app.run(debug=False)