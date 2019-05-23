import os
import sys
import flask
import io
import subprocess
import pandas as pd
from pytesseract import image_to_string
import matplotlib.pyplot as plt
from PIL import Image
print(__name__)
sys.path.append(os.path.abspath('.'))

from flask import Flask, request, jsonify, session, render_template, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename

import serve

# define the app
app = Flask(__name__)
app.secret_key = os.urandom(24)
if not os.path.exists("./uploads/"):
    os.mkdir("./uploads/");
UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
CORS(app)


# API route
@app.route('/predict', methods=['POST',  'GET'])
def predict_api():
    """API function that receives forms input and sends predictions to frontend
    """
    if request.method == 'GET':
        # Show the upload form.
        return render_template('simple_client.html')

    pdf_file = request.files['pdf_file']
    modelPath = request.form['dname']
    
    TESSERACT_CONFIG = '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1'
    image = Image.open(pdf_file)
    text_content = image_to_string(image)

    if pdf_file.filename == '':
        return '''No file selected.<a href="/predict">Click here to go back.</a>'''

    try:
        model = request.form['dname']
    except:
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''
    
    global filename
    filename = secure_filename(pdf_file.filename)

    saved_filename = os.path.join(app.config['UPLOAD_FOLDER'],filename.rsplit(".",1)[0]+".txt")
    with open(os.path.join(app.config['UPLOAD_FOLDER'],filename.rsplit(".",1)[0]+".txt"), "w") as text_file:
        text_file.write(text_content)
  
    if modelPath == "DL":
        status = subprocess.call(['./examples/clientx/predict-pipeline.sh',saved_filename])
        if status == 0:        
            df = pd.read_csv("./postprocessed/"+filename.rsplit(".",1)[0]+".csv", delimiter='~')
    else:
        status = subprocess.call(['./examples/clientx/predict-pipeline-ml.sh',saved_filename])
        if status == 0:        
            df = pd.read_csv("./postprocessed/"+filename.rsplit(".",1)[0]+".csv", delimiter='~')
    
    str_io = io.StringIO()
    df.to_html(buf=str_io, classes='table table-striped')
    html_str = str_io.getvalue()

    return '''
            <html><body>'''+html_str+'''
            <br><br><a href="/return-files">Click here to download as csv.</a><br><br>
            <a href="/predict">Click here to upload new document.</a>
            </body></html>
            '''


@app.route('/predictText', methods=['POST', 'GET'])
def predict_text_api():
    """API function that receives text input and sends predictions to frontend
    """
    if request.method == 'GET':
        # Show the enter text form.
        return render_template('full_client.html')

    sentence = request.json['input']
    modelPath = request.json['dname']

    # Empty validation
    if modelPath == '':
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''
    try:
        model = request.json['model']
    except:
        return '''No model selected.<a href="/predict">Click here to go back.</a>'''

    model_dir = modelPath + model

    app.logger.info("api_input: " + str(model_dir))
    # Get predictions for text
    model_api = serve.get_model_api1(model_dir, sentence)

    return jsonify(model_api)


@app.route('/listdir', methods=['POST', 'GET'])
def dir_listing():
    """API function that list directories.
    """
    # Joining the base and the requested path
    abs_path = request.json
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return exit(1)

    # Show directory contents
    files = os.listdir(abs_path)
    print(files)
    return jsonify(files)


@app.route('/return-files/')
def return_files_tut():
    """API function that sends predictions file to frontend.
    """
    try:
        return flask.send_file("../../postprocessed/"+filename.rsplit(".",1)[0]+".csv",attachment_filename=filename.rsplit(".",1)[0]+".csv", as_attachment=True)
    except Exception as e:
        return str(e)


@app.route('/')
def hello():
    return render_template('home.html')


@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='localhost', port=8090, debug=True)
