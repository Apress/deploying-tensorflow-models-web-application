from flask import Flask, render_template, request,flash
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import warnings

import json
import requests


warnings.simplefilter("ignore")
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_variables():
    """
    helper function to load the variables/constants once this python script has started
    """
    global headers
    headers = {"content-type": "application/json"}

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
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers) #replace the IP address with the correct one 
    predictions = json.loads(json_response.text)['predictions']

    #-----------End of prediction -----------------------------------
    predicted_class = label_names[np.argmax(predictions)]
    result = "The predicted class is : "+predicted_class
    print(result)
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
             flash('upload suitable format like jpg and png')
        os.remove(filename)
        txt = result
        return render_template("success.html", text=txt )

if __name__ == '__main__':
    load_variables() #load the model function
    app.debug = False #for production environment
    app.run(port=5000)
