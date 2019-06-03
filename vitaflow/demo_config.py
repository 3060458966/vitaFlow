import os

from vitaflow.pipeline.interfaces.utils import check_n_create

#/path/to/vitalflow_repo/
ROOT_DIR = os.path.dirname(__file__)

# Image path for processing
# Note: For Web UI display, these images are requried to avail as sub-folder of Vitaflow Annotation Tools(Flask Server)
IMAGE_ROOT_DIR = "data/receipts/"

EAST_MODEL_DIR = "vitaflow/annotate_server/static/data/east_models/east/EASTModel/exported/1558013588"

EAST_OUT_DIR = "vitaflow/annotate_server/static/data/east"

# To store cropped images - original images
BINARIZE_ROOT_DIR = "vitaflow/annotate_server/static/data/binarisation"

# To store cropped images - original images
CROPPER_ROOT_DIR = "vitaflow/annotate_server/static/data/cropped_images"

TEXT_OCR_DATA_DIR = "vitaflow/annotate_server/static/data/text_data"

TEXT_OUT_DIR = "vitaflow/annotate_server/static/data/text_out"

LOGS_DIR = "vitaflow/annotate_server/static/data/logs/"

def create_dirs():
    # create missing dir
    for each_dir in [IMAGE_ROOT_DIR,
                     EAST_OUT_DIR,
                     CROPPER_ROOT_DIR,
                     BINARIZE_ROOT_DIR,
                     TEXT_OCR_DATA_DIR,
                     TEXT_OUT_DIR,
                     LOGS_DIR]:
        each_dir = os.path.join(each_dir)
        print(each_dir)
        check_n_create(each_dir)

