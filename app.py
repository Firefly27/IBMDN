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


global graph
graph=tf.compat.v1.get_default_graph()
app=Flask(__name__)

json_path=r"./final_model/final_model.json"
json_file=open(json_path,'r')
loaded_model_json=json_file.read()
json_file.close()


h5_path=r"./final_model/final_model.h5"
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights(h5_path)
print('Model loaded.Check http://127.0.0.1:5000/')


@app.route('/')
def index():
    return render_template('digital.html')


if __name__=='__main__':
    app.run(debug=False)