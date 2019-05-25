#!flask/bin/python
import os

from flask import Flask, render_template, jsonify, request, Markup
from flask_cors import CORS

import annotate
import config
import cropper
import image_manager
import stats

image_manager.GetNewImage.refresh()

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)
# TODO: remove below line
print('Print: {}'.format(app.secret_key))

UPLOAD_FOLDER = "./static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

sample_data = {"url": "static/images/NoImage.png",
               "id": "NoImage.png",
               "folder": "collection_01/part_1",
               "annotations": [
                   {"tag": "Eagle", "x": 475, "y": 225, "width": 230.555555554, "height": 438.888888886}
               ]
               }


# ##################################### Rest API Call #################################


@app.route('/inc/validateTagsAndRegions', methods=['POST', 'GET'])
def _rest_validate_tags_and_regions():
    form_data = dict(request.form)
    # pprint(form_data)
    if 'sendInfo' in form_data.keys():
        annotate.validate_tags_and_regions(request.form)
    return _rest_get_new_image()


@app.route('/inc/getNewImage', methods=['POST', 'GET'])
@app.route('/inc/getNewImage/<image>', methods=['POST', 'GET'])
def _rest_get_new_image(image=None):
    if image:
        print('\n' + '=' * 54 + image + '\n' + '=' * 54)
        try:
            return jsonify(image_manager.GetNewImage.get_specific_image(image))
        except Exception as err:
            print("==" * 15)
            print(err)
            print(image)
            print(locals())
    image_manager.GetNewImage.refresh()
    return jsonify(image_manager.GetNewImage.get_new_image())


@app.route("/cropper/upload", methods=['POST'])
@app.route("/upload", methods=['POST'])
def _rest_cropper_upload():
    data = dict(request.form)
    print('Cropper - File upload status: {}'.format(cropper.cropper_upload(data)))
    return 'ok'


@app.route("/text_extraction_data/")
@app.route("/text_extraction_data/<start>")
@app.route("/text_extraction_data/<start>/<end>")
def _rest_page_text_extraction(start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = 10
    receipt_images = image_manager.GetNewImage.receipt_images
    data = []
    for image in list(receipt_images.keys())[start:end]:
        url = os.path.join(config.IMAGE_ROOT_DIR, image)
        text_data = 'No file found!'
        try:
            with open(os.path.join(config.TEXT_DIR, image + '.txt'), 'br') as fp:
                import unicodedata
                _data = fp.read()
                # TODO: Not able to read any files & make it Json Serializable.
                text_data = unicodedata.normalize('' + _data)
                text_data = str(text_data.encode('utf-8'))
        except:
            text_data = 'Filed to parse file!'
        data.append((url, text_data))
    html_data = jsonify(list(data))
    return html_data

# ##################################### Page Call #################################
from werkzeug import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'jpg', 'pdf', 'png'])


def allowed_filename(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/upload_file/')
def page_upload_file():
    if request.method == 'POST':
        submitted_file = request.files['file']
        if submitted_file and allowed_filename(submitted_file):
            filename = secure_filename(submitted_file.filename)
            submitted_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/<image>')
@app.route('/annotate_image')
@app.route('/annotate_image/<image>')
def page_annotate_image2(image=None):
    if image:
        print('\n' + '-' * 54 + image + '\n' + '-' * 54)
    return render_template('index.html')


# @app.route('/annotate_image')
# def page_annotate_image():
#     return render_template('index.html')


# @app.route('/review_annotation')
@app.route('/review_annotation/<image>')
def review_annotation(image):
    return render_template('index.html')


@app.route('/summary/<start>/<end>')
def _rest_show_summary(start, end):
    # TODO: setup - pagination using start and end
    start = int(start) if start.isdigit() else 0
    end = int(end) if end.isdigit() else 10
    # from random import shuffle
    receipt_images = image_manager.GetNewImage.receipt_images
    data_list = [(key, receipt_images[key]) for key in image_manager.GetNewImage.pending_images]
    data_list = sorted(data_list)[start: end]
    # shuffle(data_list)
    data_dict = dict(data_list)
    return jsonify({'receipt_images': data_dict})


# @app.route('/page_show_completed_images')
# def page_show_completed_images():
#     # Get data & show
#     # show data nicely
#     image_manager.GetNewImage.refresh()
#     # print(image_manager.GetNewImage.PendingImages)
#     return jsonify(image_manager.GetNewImage.completed_images)
#

@app.route('/show_all_images')
def page_show_all_images():
    data = jsonify([
        ('receipt_images', list(image_manager.GetNewImage.receipt_images.keys())),
        ('pending_images', list(image_manager.GetNewImage.pending_images)),
        ('completed_images', list(image_manager.GetNewImage.completed_images)),
    ])
    return render_template("stats.html", html_data=data)


@app.route('/summary')
@app.route('/summary/')
def page_show_summary():
    data = list(image_manager.GetNewImage.annotated_files.keys())
    print('Request to display {} Records'.format(len(data)))
    return render_template('summary.html', data=data)


@app.route("/cropper/<image_name>")
def page_cropper(image_name=None):
    data = {'image_name': "/" + os.path.join(config.IMAGE_ROOT_DIR, image_name)}
    return render_template("Cropper_js.html", data=data)


@app.route("/stats")
def page_stats_page():
    data = stats.get_stats()
    # print(data)
    return render_template("stats.html", html_data=Markup(data))


@app.route("/model_selection")
def page_model_selection():
    return render_template("model_selection.html")


@app.route("/text_extraction")
def page_text_extraction():
    html_data = '<p>Hello Gro</p>'
    return render_template("text_extraction.html", data=html_data)




if __name__ == '__main__':
    app.run(debug=True)  # host='172.16.49.198'
