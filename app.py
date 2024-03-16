from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.metrics import classification_report
import os

app = Flask(__name__)

model = load_model('cnn_model.h5')
model.make_predict_function()

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.
    preds = model.predict(x)
    return preds[0]

def enhance_image(img_path):
    img = Image.open(img_path)
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(1.5)
    enhanced_path = "static/enhanced_" + img_path.split("/")[-1]
    enhanced_img.save(enhanced_path)
    return enhanced_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        img_file = request.files['image']
        img_path = "static/" + img_file.filename
        img_file.save(img_path)
        
        if not os.path.exists(img_path):
            return render_template('error.html', message='File not found', image_path=img_path)
        
        enhanced_path = enhance_image(img_path)
        preds = predict(img_path)
        print(preds.shape)  # add this line to check the shape of preds
        classes = ['AGRICULTURE LAND', 'BARREN LAND', 'GRASS LAND', 'URBAN LAND']
        result = classes[np.argmax(preds)]
        fig, ax = plt.subplots()
        ax.bar(classes, preds)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Accuracy score')
        ax.set_xlabel('Class')
        plt.xticks(rotation=45)
        graph_path = "static/graph_" + img_path.split("/")[-1].split(".")[0] + ".png"
        plt.savefig(graph_path, bbox_inches='tight')
        return render_template('home.html', prediction=result, image_path=img_path, enhanced_path=enhanced_path, graph_path=graph_path)


if __name__ == '__main__':
    app.run(debug=True)