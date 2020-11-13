from flask import Flask, render_template, request,flash
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import warnings


warnings.simplefilter("ignore")
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = None

def model_load():
    """
    helper function to load the model once this python script has started
    """
    global model
    model = load_model("../model_files/keras_format/saved_model.h5") # replace with correct(absolute) path
    print('Tensorflow keras Model loaded successfully')

def preproces_input(img):
    """
    Helper function to preprocess input Image
    """
    img = Image.fromarray(img)
    img = img.resize((32, 32), Image.ANTIALIAS) #resize the image using PIL's builtin function
    img = np.array(img)
    if len(img.shape) == 2:  #if the user is uploading a black and white image
            img=np.stack((img,)*3, axis=-1)
    img = np.expand_dims(img,axis=0) # the size of the first
    img = img/255.0
    return img


def predict(img):
    img = preproces_input(img)
    # -----------Making prediction ----------------------------------
    pred = model.predict(img)
    #-----------End of prediction -----------------------------------
    predicted_class = label_names[np.argmax(pred)]
    result = "The predicted class is : "+predicted_class
    return result

app = Flask(__name__)

# Set "homepage" to index.html
@app.route('/')
def index():
    return render_template('index.html')

#Sending results to be displayed in success.html page
@app.route('/success', methods=['POST'] )
def success():
    if request.method == 'POST':
        if 'file' not in request.files: #check for file
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        filename = file.filename #retrieve the filename
        if filename == '': #if the file is empty
            flash('No selected file')
            return redirect(request.url)
        file.save(filename) #save the file
        try:
                img = plt.imread(filename) #read the input image
                result = predict(img)  #make predictions on the given image
        except:
             flash('upload suitable format like jpg and dcm')
        os.remove(filename)
        txt = result
        return render_template("success.html", text=txt )

if __name__ == '__main__':
    model_load() #load the model function
    app.debug = False #for production environment
    app.run(port=5000)
