#!flask/bin/python
import os

from flask import Flask, flash, render_template, request, redirect
from flask_cors import CORS
from werkzeug import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)
# TODO: remove below line
print('Print: {}'.format(app.secret_key))

# UPLOAD_FOLDER = "static/data/uploads/"
UPLOAD_FOLDER = "static/data/preprocess/"

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'jpg', 'pdf', 'png'])


def allowed_filename(filename):
    print("Checking file extension validation for {}".format(filename))
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/upload_file/', methods=['GET', 'POST'])
def page_upload_form(filename=None):
    if request.method == 'GET':
        if filename:
            print('printing {}'.format(filename))
        return render_template('demo.html')
    if request.method == 'POST':
        # check if the post request has the file part
        print('POST')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_filename(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded!! Processing will start soon, please check "Sumary" page for updates')
            print('File successfully uploaded {}'.format(file_path))
            # Run pipeline - daemon job
            run_pipeline(filename)
            # return redirect('/upload_file', filename=file.filename)
            return redirect('/upload_file/')


@app.route('/logs/<filename>')
def show_logs(filename):
    from glob import glob
    fmt = 'static/data/logs/*{}*'.format(filename)
    files = glob(fmt)
    if files:
        data = open(files[0]).read()
    else:
        data = 'please check, no logs found!!'
    return """<html><body>
    <pre>{}</pre>
    <body></html>""".format(data)


def show_uploaded_images():
    from glob import glob
    html_data = ''
    for url in glob(UPLOAD_FOLDER + '*.jpg'):
        filename = url.split('/')[-1]
        html_data += '<li> <b>{}</b><ul>' \
                     '<li> Input Image: <a href="/{}">{}</a> </li>' \
                     '<li> Outpage: <a href="/uploads/{}">Link</a> </li>' \
                     '<li> Logs: <a href="/logs/{}">Link</a> </li>' \
                     '</ul></li>   '.format(filename, url, filename, filename, filename)
    html_data = "<html><body><ul>{}<ul></body></html>".format(html_data)
    from flask import Markup
    return render_template('demo_result.html', html_data=Markup(html_data), data=None)


@app.route('/uploads/<filename>')
def show_uploaded_image_details(filename):
    import config
    from glob import glob
    image_data = "/{}/{}".format(UPLOAD_FOLDER, filename)
    bin_data = glob(os.path.join(config.BINARIZE_ROOT_DIR, '*' + filename))
    if bin_data:
        bin_data = bin_data[0]
    else:
        bin_data = ''
    text_data = glob(os.path.join(config.TEXT_IMAGES, filename.rsplit('.', 1)[-2] + '/*'))
    text_data.sort()
    text_data_images = sorted([_ for _ in text_data if '.png' in _],
                              key=lambda fn: int(fn.rsplit('/')[-1].split('.')[0]))
    # Tesseract Text
    f_name = os.path.join(config.TEXT_DIR, filename.split('.')[0] + '.tesseract.txt')
    print(f_name)
    text_data_tesseract = open(f_name).read() if os.path.isfile(f_name) else ""
    # Calamari Text
    f_name = os.path.join(config.TEXT_DIR, filename.split('.')[0] + '.pred.txt')
    text_data_calamari = open(f_name).read() if os.path.isfile(f_name) else ""
    print(bin_data)
    data = {
        'image_data': image_data,
        'binarisation': bin_data,
        'text2Lines': text_data_images,
        'tesseract': text_data_tesseract,
        'calamari': text_data_calamari
    }
    return render_template('demo_result.html', html_data=None, data=data)


@app.route('/uploads/')
def page_show_uploads():
    return show_uploaded_images()


def run_pipeline(filename=None):
    print('Running East Pipeline')
    import os
    command = 'cd ../.. && make east binarisation crop2box tesseract calmari text2file'
    if filename:
        # https://www.cyberciti.biz/faq/redirecting-stderr-to-stdout/
        command = command + ' &>vitaflow/annotate_server/static/data/logs/{}.log &'.format(
            filename)
        print('Running East Pipeline, Logs are at vitaflow/annotate_server/static/data/logs/{}.log &'.format(filename))
    os.system(command)

if __name__ == '__main__':
    app.run(debug=True)  # host='172.16.49.198'
