import os

from bin.utils import check_n_create

ROOT_DIR = os.path.dirname(__file__)

# Image path for processing
# Note: For Web UI display, these images are requried to avail as sub-folder of Vitaflow Annotation Tools(Flask Server)
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

# Testing Purposes
DUMMY_DIR = "static/data/DUMMY"

# Tesseract Config
TESSERACT_CONFIG = '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1'

# Collection name
# COLLECTION_NAME = "collection_01"

# Not annotated image 80% to be presented to user
# ratio_new_old = 80

# Acceptable file extension
IMAGE_EXTS = ['.JPG', '.jpg', '.png']

# Acceptable file extension
PDF_EXTS =['.pdf']

# Time inverval to re-check images
OS_FILE_REFRESH_TIME_INVTERVAL = 2 * 60  # Shift to inotify

# Acceptable txt file extension
OCR_TEXT_EXTS = ['.pred.txt','.tesseract.txt']

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

# Output of
PDF2TEXT_DIR="static/data/text"

# TODO: Update following acc to ocr_calamari
CALAMARI_MODELS = ['static/data/calamari_models/model_00037200.ckpt',
                   'static/data/calamari_models/model_00045600.ckpt',
                   'static/data/calamari_models/model_00114400.ckpt',
                   'static/data/calamari_models/model_00123200.ckpt',
                   'static/data/calamari_models/model_00131400.ckpt']

CALAMARI_MODELS = [os.path.join(ROOT_DIR, _) for _ in CALAMARI_MODELS]
