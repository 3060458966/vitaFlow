#!flask/bin/python
import os

from flask import Flask, flash, render_template, request, redirect
from flask_cors import CORS
from werkzeug import secure_filename

import image_manager

image_manager.GetNewImage.refresh()

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)
# TODO: remove below line
print('Print: {}'.format(app.secret_key))

UPLOAD_FOLDER = "static/data/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'jpg', 'pdf', 'png'])


def allowed_filename(filename):
    print("Checking file extension validation for {}".format(filename))
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# @app.route('/', methods=['GET', 'POST'])
# @app.route('/upload_file/', methods=['GET', 'POST'])
# def page_upload_file():
#     print('--' * 15)
#     if request.method == 'POST':
#         submitted_file = request.files['file']
#         print('--' * 15)
#         if submitted_file and allowed_filename(submitted_file.filename):
#             filename = secure_filename(submitted_file.filename)
#             print('Saving File at {}'.format(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
#             submitted_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('page_upload_file', filename=filename))
#
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form action="" method=post enctype=multipart/form-data>
#       <p><input type=file name=file>
#          <input type=submit value=Upload>
#     </form>
#     '''

@app.route('/')
@app.route('/upload_file/', methods=['GET'])
@app.route('/upload_file/<filename>')
def page_upload_form(filename=None):
    if filename:
        print('printing {}'.format(filename))
    return render_template('demo.html')


@app.route('/upload_file/', methods=['POST'])
@app.route('/upload_file/', methods=['POST'])
def page_upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_filename(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File(s) successfully uploaded')
            return redirect('/upload_file', filename=file.filename)


def show_uploaded_images():
    from glob import glob
    html_data = ''
    for url in glob(UPLOAD_FOLDER + '*'):
        filename = url.split('/')[-1]
        html_data += '<li><a href="/{}">{}</a>     <a href="/uploads/{}">ProcessingDetails</a>      </li>   '.format(
            url, filename, filename)
    html_data = "<html><body><ul>{}<ul></body></html>".format(html_data)
    return html_data


@app.route('/uploads/<filename>')
def show_uploaded_image_details(filename):
    html_data = []
    html_data.append('<li><img src="/{}/{}"></li>'.format(UPLOAD_FOLDER, filename))
    html_data.append("<html><body><ul>{}<ul></body></html>".format(html_data))
    return '\n'.join(html_data)


@app.route('/uploads/')
def page_show_uploads():
    return show_uploaded_images()


if __name__ == '__main__':
    app.run(debug=True)  # host='172.16.49.198'
