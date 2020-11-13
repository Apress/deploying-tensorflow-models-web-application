# Apress-Deploying TensorFlow models to WebApplications
Repository containing all the codes for Apress course on deploying TensorFlow models to a web application 

## Code requirements : 
1) Python 3.6 or higher 
2) TensorFlow 2.0 or higher 
3) Docker (Required for TensorFlow Serving API) 
4) Stand alone TFLite interpreter (Optional,  Install your version https://www.tensorflow.org/lite/guide/python ) 



## Notes:
1) TensorFlow JavaScript : To avoid cross-origin errors, place your model files in the same model and modify your address and port accordingly 'http://127.0.0.1:8000/model_files/model.json'
2) FlaskAPI, FlaskTFLite : Be sure to modify the path of your model 
3) TF Serve address: If you are serving your model in a remote server, replace "localhost" with the IP address in app.py.
