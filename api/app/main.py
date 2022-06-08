# https://github.com/avinassh/pytorch-flask-api-heroku

import io
import os
import sys
from time import time
from urllib import request

from PIL import Image
from flask import Flask, send_file
from flask import request, render_template
from werkzeug.utils import redirect

sys.path.append("C:\\Users\\Mathias\\Documents\\Projets_Python\\image_manager\\src\\image_manager\\super_resolution")

from super_resolution import get_prediction

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
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
    start = time()
    sr_image_path = get_prediction(lr_image_path)
    print(f"Prediction run in {time() - start:.2f} s.", flush=True)

    return render_template('result.html', sr_image_path=sr_image_path, lr_image_path=lr_image_path)


@app.route('/download/<filepath>', methods=['GET'])
def download_image(filepath):
    """
    Download a super resolution image.
    """
    sr_image_path = request.view_args['filepath']
    return send_file(sr_image_path,
                     mimetype='image/png',
                     attachment_filename=os.path.basename(sr_image_path),
                     as_attachment=True)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", debug=True, port=int(os.environ.get('PORT', 5000)))  # Set debug true to load reload server auto on changes
    app.run(host="0.0.0.0")  # Set debug true to load reload server auto on changes
