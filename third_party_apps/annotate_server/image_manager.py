import os
import random
import time

import tqdm

from vitaflow.pipeline.preprocessor import binarisation
import config
from vitaflow.pipeline.interfaces.datatypes import Singleton
from vitaflow.pipeline.interfaces.utils import get_folder_config
from vitaflow.pipeline.interfaces.utils import trim_file_ext

# from pprint import pprint

preference_list = [
    'binarisation_url',
    'cropper_url',
    'url'
]


def get_image_url(image_dict):  # , pref_list=preference_list):
    for key in preference_list:
        url = image_dict.get(key)
        if url:
            return url
    raise Exception('No Preference List\'s URL is in {}'.format(str(image_dict)))


def _generate_binarisation_paths(image_id):
    receipt_images = GetNewImage.receipt_images
    image_path = os.path.join(config.ROOT_DIR, receipt_images[image_id]['url'])
    binarisation_path = os.path.join(config.ROOT_DIR, config.BINARIZE_ROOT_DIR, receipt_images[image_id]['file'])
    binarisation_image_root_path = os.path.join(config.BINARIZE_ROOT_DIR, receipt_images[image_id]['file'])
    return image_path, binarisation_path, binarisation_image_root_path


def _generate_binarisation_images(image_id):
    (image_path, binarisation_path, binarisation_image_root_path) = _generate_binarisation_paths(image_id)
    if not os.path.isfile(binarisation_path):
        binarisation.binarisation(image_path, binarisation_path)
        # print('Binarisation done for: {}'.format(image_id))
    # else:
    # print('Binarisation found for: {}'.format(image_id))
    return binarisation_image_root_path


def _regenerate_binarisation_images(image_id):
    (image_path, binarisation_path, binarisation_image_root_path) = _generate_binarisation_paths(image_id)
    # regeneration is due to Cropper
    image_path = os.path.join(config.ROOT_DIR, GetNewImage.receipt_images[image_id]['cropper_url'])

    print('os- remove- files {} - {}'.format(binarisation_path, os.remove(binarisation_path)))
    # if not os.path.isfile(binarisation_path):
    #     binarisation.binarisation(image_path, binarisation_path)
    #     print('Binarisation done for: {}'.format(image_id))
    # else:
    print('\n' * 5)
    print((image_path, binarisation_path))
    binarisation.binarisation(image_path, binarisation_path)
    print('Binarisation redone for: {}'.format(image_id))
    return binarisation_image_root_path


def gen_cropper_binarisation(image_name):
    image_id = trim_file_ext(image_name)
    _regenerate_binarisation_images(image_id)
    return 'ok - _generate_binarisation_images'


# noinspection PyCompatibility
class GetNewImage(metaclass=Singleton):
    # input images
    receipt_images = {}
    # features - cropping & rotation
    cropper_images = {}
    # feature - bin_images
    bin_images = {}
    # output files
    annotated_files = {}
    # other
    pending_images = []
    completed_images = []
    _last_refresh_ = 0
    _refresh_interval_ = config.OS_FILE_REFRESH_TIME_INVTERVAL
    _default_image_ = "/static/images/NoImage.png"  # TODO: Remove this - set this at UI - JS Level

    @staticmethod
    def get_default_request_response():
        send_info = {"url": GetNewImage._default_image_,
                     "id": os.path.basename(GetNewImage._default_image_),
                     "folder": "",
                     "annotations": []}
        return send_info

    @staticmethod
    def get_specific_image(image_file):
        image_name = trim_file_ext(image_file)

        send_info = {
            'id': GetNewImage.receipt_images[image_name]['file'],
            # 'url': GetNewImage.receipt_images[image_name]['url'],
            'url': get_image_url(GetNewImage.receipt_images[image_name]),
            'folder': '',
            "annotations": []
        }
        return send_info

    @staticmethod
    def get_old_image():
        if GetNewImage.annotated_files:
            image_file = random.choice(list(GetNewImage.annotated_files.keys()))
            send_info = GetNewImage.get_specific_image(image_file)
            # TODO - Add Annotations
        else:
            send_info = GetNewImage.get_default_request_response()
        return send_info

    @staticmethod
    def get_new_image():
        if GetNewImage.pending_images:
            image_file = random.choice(GetNewImage.pending_images)
            send_info = GetNewImage.get_specific_image(image_file)
        else:
            send_info = GetNewImage.get_default_request_response()
        print('GetNewImage returned {}'.format(send_info['id']))
        return send_info

    @staticmethod
    def _get_images():
        images_dict = get_folder_config(os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR),
                                        config.IMAGE_EXTS,
                                        config.ROOT_DIR)
        return images_dict

    @staticmethod
    def _get_annotations():
        xml_dict = get_folder_config(os.path.join(config.ROOT_DIR, config.ANNOTATIONS_DIR),
                                     ['.xml'],
                                     config.ROOT_DIR)
        return xml_dict

    @staticmethod
    def update_data(image=None):
        image_key = trim_file_ext(image)
        if image_key in GetNewImage.pending_images:
            GetNewImage.pending_images.remove(image_key)
            GetNewImage.completed_images.append(image_key)
            del GetNewImage.receipt_images[image_key]
            print('Image Annotations Done for {}'.format(image))

    @staticmethod
    def refresh(image=None):
        if image:
            GetNewImage.update_data(image)
        if time.time() - GetNewImage._last_refresh_ < GetNewImage._refresh_interval_:
            print('GetNewImage is still fresh!!')
            return

        GetNewImage._last_refresh_ = time.time()
        print('GetNewImage is refreshed!!')
        GetNewImage.annotated_files = GetNewImage._get_annotations()
        GetNewImage.receipt_images = GetNewImage._get_images()

        xml_files_keys = set(GetNewImage.annotated_files.keys())
        image_files_keys = set(GetNewImage.receipt_images.keys())
        GetNewImage.pending_images = list(image_files_keys - xml_files_keys)
        GetNewImage.completed_images = list(image_files_keys.intersection(xml_files_keys))

        # feature
        GetNewImage.gen_binarisation_images()
        GetNewImage._get_cropper_data()
        GetNewImage._get_binarisation_data()

        # pprint
        # pprint(GetNewImage.receipt_images)

    @staticmethod
    def _get_cropper_data():
        # fetch cropper data
        cropper_images = \
            get_folder_config(config.CROPPER_ROOT_DIR,
                              file_exts=config.IMAGE_EXTS,
                              trim_path_prefix=config.ROOT_DIR)

        receipt_images = GetNewImage.receipt_images
        # update cropper data
        for key, key_value in cropper_images.items():
            receipt_images[key]['cropper_url'] = key_value['url']

    @staticmethod
    def _get_binarisation_data():
        # fetch cropper data
        binarisation_images = \
            get_folder_config(config.BINARIZE_ROOT_DIR,
                              file_exts=config.IMAGE_EXTS,
                              trim_path_prefix=config.ROOT_DIR)

        receipt_images = GetNewImage.receipt_images
        # update cropper data
        for key, key_value in tqdm.tqdm(binarisation_images.items(), desc='Binarise images Url Tagging'):
            receipt_images[key]['binarisation_url'] = key_value['url']

    @staticmethod
    def update_binarisation_data(image_file):
        image_id = trim_file_ext(image_file)
        GetNewImage.receipt_images[image_id]['binarisation_url'] = os.path.join(config.BINARIZE_ROOT_DIR, image_file)

    @staticmethod
    def update_cropper_data(image_file):
        image_id = trim_file_ext(image_file)
        GetNewImage.receipt_images[image_id]['cropper_url'] = os.path.join(config.CROPPER_ROOT_DIR, image_file)

    @staticmethod
    def gen_binarisation_images():
        all_receipt_images = list(GetNewImage.receipt_images.keys())
        for image_id in tqdm.tqdm(all_receipt_images, desc='Gen Binarise images'):
            GetNewImage.receipt_images[image_id]['binarisation_url'] = _generate_binarisation_images(image_id)
