# server.py

import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))  # Specify the destination folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload_photo.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return redirect(request.url)
    photo = request.files['photo']
    if photo.filename == '':
        return redirect(request.url)
    if photo:
        # Save the uploaded file to the destination folder
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
        return f'Photo "{photo.filename}" uploaded successfully!'
    return 'Error uploading photo.'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
