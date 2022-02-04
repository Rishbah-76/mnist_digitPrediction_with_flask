from flask import Flask, render_template, request,redirect,url_for,flash,send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
import tensorflow as tf

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER     
def model_predict(img_path, model):
    print(img_path)
    #img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    #x=x/255
    #x = np.expand_dims(x, axis=0)
   
   #prediction model
    img=cv2.imread(img_path)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #loading the imag as gray
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #prediction in pred variable
    pred3=model.predict(gray_image.reshape(1,784))

    #result is predargmax
    #preds=pred3.argmax()

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)
    
    #preds = model.predict(x)
    
    preds=np.argmax(pred3, axis=1)
    if preds==0:
        preds="Image Predicted as: 0"
    elif preds==1:
        preds="Image Predicted as: 1"
    elif preds==2:
        preds="Image Predicted as: 2"
    elif preds==3:
        preds="Image Predicted as: 3"
    elif preds==4:
        preds="Image Predicted as: 4"
    elif preds==5:
        preds="Image Predicted as: 5"
    elif preds==6:
        preds="Image Predicted as: 6"
    elif preds==7:
        preds="Image Predicted as: 7"
    elif preds==8:
        preds="Image Predicted as: 8"
    else:
        preds="Image Predicted as: 9"
    return preds

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS     
        
"""
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

"""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload', name=filename))
    return render_template("index.html")


@app.route('/predict/<name>')
def upload(name):
        # Get the file from post request
        img_name=(os.path.join(app.config['UPLOAD_FOLDER'], name))
            #return redirect(url_for('download_file', name=filename))
        # Make prediction
        model=tf.keras.models.load_model("model1.h5")
        preds = model_predict(img_name, model)
        result=preds
        return render_template("predict.html", result=result)

if __name__ == '__main__':
    app.run(port=5001,debug=True)

    '''
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)'''