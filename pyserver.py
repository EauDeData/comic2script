from io import BytesIO
from flask import Flask, render_template, request, send_from_directory, send_file
import os
import warnings
import json
import numpy as np
from PIL import Image
import recognition
import database
import cv2

html_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
app = Flask(__name__, template_folder=html_dir+'html')
warnings.warn("Be careful! This server isn't protected ad therefore should only be used in debugging; never in production.")

all_comics = database.list_comics('/home/adri/Desktop/cvc/data/comics/comicbookplus_data/')

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/<path:path>/<file>.jpg')
def file(path, file, extension):
    return send_from_directory('/', f"{path}/{file}.jpg")


@app.route('/volumes/<value>/')
def volumes_index(value):
    value = '?' + value
    editons = list(all_comics[value].keys())
    return render_template(f"editions.html", vols = [x.replace('?', '') for x in editons], string = value[1:])

@app.route('/volumes/<value>/edition/<value2>')
def pages(value, value2):
    value = '?' + value
    value2 = '?' + value2
    pages = all_comics[value][value2]
    return render_template(f"pages.html", vols = [x.replace('.jpg', '') for x in pages], string = value[1:], value2 = value2[1:])

@app.route('/volumes/<value>/edition/<value2>/page/<num>')
def return_page(value, value2, num):

    default_  = f'/home/adri/Desktop/cvc/data/comics/comicbookplus_data/?{value}/?{value2}/0/{num}.jpg'
    im, mask, cont, _, _ = recognition.layout_contours(default_)

    if len(im.shape) != 3: im = cv2.cvtColor(~im, cv2.COLOR_GRAY2RGB)
    final = 0.7 * im + 0.3 * mask + cont
    final = (final - final.min()) / (final.min() - final.max()) * 255

    final = final.astype(np.uint8)

    return serve_pil_image(Image.fromarray(final))

@app.route('/example')
def example():
    return render_template('example.html')


@app.route('/')
def index():
    volumes = list(all_comics.keys())
    return render_template(f"volumes.html", vols = [x.replace('?', '') for x in volumes])

ip = "0.0.0.0:5000"
ip, port = ip.split(':')

app.run(host=ip, port = int(port[:]), debug = True)