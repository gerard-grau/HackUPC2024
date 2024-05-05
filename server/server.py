from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request, redirect, url_for
# from search_functions import show_product_neighbours

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))  # Specify the destination folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + '/static/'

@app.route('/upload_photo')
def index():
    return render_template('upload_photo.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/nearest_clothes')
def nearest_clothes():
    return render_template('nearest_clothes.html')

@app.route('/upload', methods=['GET', 'POST'])
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
        photo.filename = "image_to_predict"
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
        return render_template('display_photo.html')
    return 'Error uploading photo.'

@app.route('/display_photo')
def display_photo():
    return render_template('display_photo.html')


@app.route('/search', methods=['POST'])
def search():
    article_path = request.form['article-name']
    # call python function using article_path
    # store the image in the static/images folder
    return render_template('nearest_clothes.html')



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)