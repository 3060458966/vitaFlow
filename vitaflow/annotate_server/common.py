import os

from vitaflow.pipeline.interfaces import utils


def verify_isfile(full_path_file_name):
    return os.path.isfile(full_path_file_name)


def verify_isimagefile(full_path_file_name, exts=['.JPG', '.jpg', '.png']):
    return utils.get_file_ext(full_path_file_name) in exts
