import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
# import CNN
import numpy as np
import torch
import tensorflow as tf
import keras
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')



model=tf.keras.models.load_model('7')
CLASS_NAMES = ["Pepper_bell_Bacterial_spot", "Pepper_bell_healthy", "Potato_Early_blight", "Potato_Late_Blight", "Potato_Healthy", "Tomato_Bacterial_spot","Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato_Target_Spot","Tomato_Tomato_YellowLeaf_Curl_Virus","Tomato_Tomato_mosaic_virus","Tomato_healthy"]

def prediction(image_path):
    image = Image.open(image_path)
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': (round(float(confidence),4))*100,
        'sno':np.argmax(predictions[0])
    }


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        
        #  disease_info
        sno1=pred['sno']
        confidence=pred['confidence']
        title =disease_info['disease_name'][sno1]
        description =disease_info['description'][sno1]
        # confidence=disease_info['confidence'][sno1]
        prevent = disease_info['Possible Steps'][sno1]
        image_url = disease_info['image_url'][sno1]
        supplement_name = supplement_info['supplement name'][sno1]
        supplement_image_url = supplement_info['supplement image'][sno1]
        supplement_buy_link = supplement_info['buy link'][sno1]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , confidence=confidence,
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
        # return render_template('submit.html' , title = title ,  pred = pred , desc = description, prevent = prevent , 
        #                        image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
