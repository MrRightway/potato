from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#%%

app = Flask(__name__)

#%%

model_path="C:/data science/Potato  Work Internship/Using Flask/model/potatoes.h5"
model = load_model(model_path)
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')



#%%
def predict(model, img_path):
    class_name = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class

#%%
# img_path="C:/data science/Potato  Work Internship/Using Flask/dataset/Potato___Late_blight/0acdc2b2-0dde-4073-8542-6fca275ab974___RS_LB 4857.JPG"
# a=predict(model,img_path)
# print(a)
#%%

class_name=['Potato  Early Blight', 'Potato Late blight', 'Potato healthy']



#%%

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

#%%
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)

        # Make prediction
        preds = predict(model ,file_path)

        # Process your result for human
        
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)