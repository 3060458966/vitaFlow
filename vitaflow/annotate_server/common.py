import os

import config
from bin import utils


def verify_input_file(file_with_fullpath):
    return os.path.isfile(file_with_fullpath)


def verify_image_ext(file_with_fullpath):
    return utils.get_file_ext(file_with_fullpath) in config.IMAGE_EXTS
