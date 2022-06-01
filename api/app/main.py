# https://github.com/avinassh/pytorch-flask-api-heroku

import io
import os
import sys
from urllib import request

from PIL import Image
from flask import Flask
from flask import request, render_template
from werkzeug.utils import redirect

sys.path.append("C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution")
sys.path.append("..\\.\\")

from super_resolution import get_prediction

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  #TODO make a separate predict route
def predict():
    """
    Make prediction for a file.
    """
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files.get('file')
    if not file:
        return

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    lr_image_path = os.path.join("static", "images", file.filename)
    image.save(lr_image_path)
    sr_image_path = get_prediction(lr_image_path)

    return render_template('result.html', sr_image_path=sr_image_path, lr_image_path=lr_image_path)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))  # Set debug true to load reload server auto on changes
