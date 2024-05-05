# server.py
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))  # Specify the destination folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_photo')
def index():
    return render_template('upload_photo.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results_picture', methods=['GET', 'POST'])
def upload_photo():
    file_key = 'photo' if 'photo' in request.files else 'gallery_photo'
    if file_key not in request.files:
        return redirect(request.url)
    photo = request.files[file_key]
    if photo.filename == '':
        return redirect(request.url)
    if photo:
        # Save the uploaded file to the destination folder
        filename = secure_filename(photo.filename)
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
        return render_template('display_photo.html')
    return 'Error uploading photo.'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)