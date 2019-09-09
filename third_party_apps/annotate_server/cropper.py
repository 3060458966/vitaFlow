"""

Rules:
    1. One cannot modify image once Annotated.

TODO:
    1. Summary of all Cropped Images
    2.

Annotation Tools:
    1. Extra buttons
        1. Reload Image
        2. Goto Cropper Image
    2.
"""

import base64
import os
import pickle
from shutil import copyfile

import config
import image_manager


def cropper_upload(data):
    # input data is dict with key - image_name, image_base64_data
    image_name = data['fileName'][0]
    image_base64_data = data['fileToUpload'][0]
    print('Cropper Upload {}'.format(image_name))
    if image_name:
        # Generate cropper & binarisation image
        gen_cropper_file(image_name, image_base64_data)
        image_manager.GetNewImage.update_cropper_data(image_name)
        image_manager.gen_cropper_binarisation(image_name)
        image_manager.GetNewImage.update_binarisation_data(image_name)
    return 'ok'


def gen_cropper_file(image_name, image_base64_data):
    # print(str(image_base64_data)[:100])
    # verify @ https://codebeautify.org/base64-to-image-converter#
    print(locals())
    original_file = os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR, image_name)
    image_name = os.path.join(config.ROOT_DIR, config.CROPPER_ROOT_DIR, image_name)
    if not os.path.isfile(original_file):
        copyfile(original_file, image_name)
        image_name = original_file
        print('Taking a backup copy of the Image')
    try:
        data = image_base64_data.split(',')[-1]
        with open(image_name, 'bw') as fp:
            fp.write(base64.b64decode(data))
        print('Saving file to Image {}'.format(image_name))
    except Exception as err:
        print(err)
        image_name += '.pk'
        with open(image_name, 'bw') as fp:
            pickle.dump(str(image_base64_data), fp)
        print('Error: Using pickle !!')
        print('Saving file to Image {}'.format(image_name))


def get_new_image():
    pass


def show_cropped_images():
    pass
