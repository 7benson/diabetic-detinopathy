import os
from os import path, walk
from image_processing import extractBloodVessels, extractExudates
from flask import Flask,render_template,redirect, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

import os.path

EXTRACTED_EXUDATES_OUTPUT="./static/output/extracted-exudates.png"
EXTRACTED_BLOOD_VESSELS_OUTPUT='./static/output/extracted-vessels.png'

IMG_HEIGHT = 32
IMG_WIDTH = 32

channels = 3
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app_data = {
    "name":         "Diabetic Retinopathy Detection",
    "description":  "Digital Image Processing Final Project",
    "author":       "Jayanth Kumar & Ravi Kiran",
    "html_title":   "Diabetic Retinopathy Detection",
    "project_name": "Traffic Sign Classifier",
    "keywords":     "Diabetic, Retinopathy ,Detection"
}
@app.route('/')
def home():
    return render_template('index.html', app_data=app_data)

@app.route('/predict/', methods = ['GET', 'POST'])
def upload_image():
    if len(request.files) ==0:
        return render_template('home.html', app_data=app_data)
    print(request.files)
    file = request.files['file']
    print()
    if file.filename == '':
        print('No image selected for uploading')
        return render_template('home.html', app_data=app_data)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('Image successfully uploaded and displayed below')

        image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(image_path)
        extractBloodVessels(image_path)
        extractExudates(image_path)
        image_path="."+image_path

        return render_template('predict.html',image_path=image_path,backgroundImage=request.url_root+"static/images/slide-01.jpg"
                                ,inputImageAddr=request.url_root+image_path[3:],extracted_exudates=request.url_root+EXTRACTED_EXUDATES_OUTPUT,extracted_blood_vessels=request.url_root+EXTRACTED_BLOOD_VESSELS_OUTPUT)
    else:
        return render_template('index.html', app_data=app_data)

extra_dirs = ['./static/styles','./static/js','./templates']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

if __name__ == '__main__':
    app.run(debug=True,extra_files=extra_files)