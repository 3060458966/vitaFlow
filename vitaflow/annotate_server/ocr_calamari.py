'''
Demo sample example of how to include calamari_ocr into python code
'''
import os
from glob import glob

from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import run as calamari_ocr_run

import config
from bin.plugin import TextExtPluginModel

# TODO:
# - convert to gray scale images
# - fixed width input 42 px


# Add your files here
calamari_models = [
# 'vitaflow/annotate_server/static/data/calamari_models/model_00037200.ckpt',
#                    'vitaflow/annotate_server/static/data/calamari_models/model_00045600.ckpt',
#                    'vitaflow/annotate_server/static/data/calamari_models/model_00114400.ckpt',
#                    'vitaflow/annotate_server/static/data/calamari_models/model_00123200.ckpt',
#                    'vitaflow/annotate_server/static/data/calamari_models/model_00131400.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00117200.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00132600.ckpt']

calamari_input_images = glob(os.path.join(config.ROOT_DIR, config.TEXT_IMAGES) + '/*/*')  # Add your files here


class args:
    batch_size = 1
    checkpoint = calamari_models
    dataset = DataSetType.FILE
    extended_prediction_data = False
    extended_prediction_data_format = 'json'
    files = calamari_input_images
    no_progress_bars = False
    output_dir = None
    pagexml_text_index = 1
    processes = 1
    text_files = None
    verbose = False
    voter = 'confidence_voter_default_ctc'


def main(source_file, destination_file=None):
    args.files = [source_file]
    calamari_ocr_run(args)


def main_parallel(source_files):
    args.files = source_files
    calamari_ocr_run(args)


class OCR_Calamari(TextExtPluginModel):
    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.TEXT_IMAGES
        self.destination_folder = config.TEXT_IMAGES
        # Transformation function for converting source_image to destination_image
        self.operator_func = main
        self.parallel_operator_func = main_parallel


if __name__ == '__main__':
    tt = OCR_Calamari()
    tt.plugin_inputs()
    tt.bulk_run()
