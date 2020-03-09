# import the necessary packages
import os
import ssl
import sys
import traceback
from os import path
from pathlib import Path

import fastai
import torch
from flask import Flask, jsonify, render_template, request, send_file, flash, redirect
from werkzeug.utils import secure_filename

from app_utils import (clean_all, clean_me, convertToJPG, create_directory,
                       download, generate_random_filename, get_model_bin)
from deoldify.visualize import *

torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ALLOWED_EXTENSIONS = {
    'avi', 'mov', 'flv', 'swf', 'webm', 'ts', 'mpeg', 'mpv', 'mp4', 'png',
    'jpg', 'jpeg', 'gif', 'jfif'
}

VIDEO_FORMATS = {
    'avi', 'mov', 'flv', 'swf', 'ts', 'webm', 'mpeg', 'mpv', 'mp4'
}
app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_FORMATS


@app.route('/')
def home():
    return render_template('index.html')


# define a predict function as an endpoint
@app.route("/process", methods=["POST", 'GET'])
def process():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    submitted_file = request.files['file']
    if submitted_file.filename == '':
        flash('No selected file')
        return redirect('/')
    if submitted_file and allowed_file(submitted_file.filename):
        filename = secure_filename(submitted_file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                  'deoldified_' + filename)

        submitted_file.save(input_path)
        render_factor = int(request.form.get('render_factor', 10))
        try:
            if is_video(input_path):
                output_path = os.path.join(results_video_directory,
                                           os.path.basename(input_path))
                return process_video(input_path, output_path, render_factor)
            output_path = os.path.join(results_img_directory,
                                       os.path.basename(input_path))
            return process_image(input_path, output_path, render_factor)
        finally:
            clean_all([input_path, output_path])
    return redirect('/')


def process_image(input_path, output_path, render_factor):
    try:
        try:
            image_colorizer.plot_transformed_image(path=input_path,
                                                   figsize=(20, 20),
                                                   render_factor=render_factor,
                                                   display_render_factor=True,
                                                   compare=False)
        except:
            convertToJPG(input_path)
            image_colorizer.plot_transformed_image(path=input_path,
                                                   figsize=(20, 20),
                                                   render_factor=render_factor,
                                                   display_render_factor=True,
                                                   compare=False)

        callback = send_file(output_path,
                             as_attachment=True,
                             mimetype='image/jpeg')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


def process_video(input_path, output_path, render_factor):
    try:
        video_path = video_colorizer.colorize_from_file_name(
            file_name=input_path, render_factor=render_factor)
        callback = send_file(output_path,
                             as_attachment=True,
                             mimetype='application/octet-stream')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


if __name__ == '__main__':
    global upload_directory
    global results_img_directory
    global image_colorizer

    upload_directory = 'data/upload/'
    create_directory(upload_directory)

    results_img_directory = './result_images'
    create_directory(results_img_directory)

    results_video_directory = './video/result/'
    create_directory(results_video_directory)

    model_directory = './models'
    create_directory(model_directory)

    artistic_model_url = 'https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeArtistic_gen.pth?dl=1'
    get_model_bin(artistic_model_url,
                  os.path.join(model_directory, 'ColorizeArtistic_gen.pth'))

    image_colorizer = get_image_colorizer(artistic=True)

    video_model_url = 'https://www.dropbox.com/s/336vn9y4qwyg9yz/ColorizeVideo_gen.pth?dl=1'
    get_model_bin(video_model_url,
                  os.path.join(model_directory, 'ColorizeVideo_gen.pth'))

    video_colorizer = get_video_colorizer()

    port = 5000
    host = '0.0.0.0'
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['UPLOAD_FOLDER'] = upload_directory
    app.run(host=host, port=port, threaded=False)
