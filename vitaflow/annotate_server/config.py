import os

from bin.utils import check_n_create

ROOT_DIR = os.path.dirname(__file__)

# Image path for processing
IMAGE_ROOT_DIR = "static/data/images"

# To store cropped images - original images
CROPPER_ROOT_DIR = "static/data/cropper"

# To store cropped images - original images
BINARIZE_ROOT_DIR = "static/data/binarisation"

# To store annotation xml files
ANNOTATIONS_DIR = "static/data/annotations"

# EAST IMAGES
EAST_DIR = "static/data/east"

# DOC2TEXT IMAGES
DOC2TEXT_DIR = "static/data/preprocess"

# To store text images
TEXT_IMAGES = "static/data/text_images"

# To store annotation xml files
TEXT_DIR = "static/data/text_data"

# Tesseract Config

TESSERACT_CONFIG = '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1'

# Collection name
# COLLECTION_NAME = "collection_01"

# Not annotated image 80% to be presented to user
# ratio_new_old = 80

# Acceptable file extension
IMAGE_EXTS = ['.jpg', '.jpg']

# Time inverval to re-check images
OS_FILE_REFRESH_TIME_INVTERVAL = 2 * 60  # Shift to inotify

# create missing dir
for each_dir in [IMAGE_ROOT_DIR,
                 CROPPER_ROOT_DIR,
                 BINARIZE_ROOT_DIR,
                 ANNOTATIONS_DIR,
                 TEXT_DIR,
                 TEXT_IMAGES,
                 DOC2TEXT_DIR
                 ]:
    each_dir = os.path.join(ROOT_DIR, each_dir)
    check_n_create(each_dir)
