'''
Demo sample example of how to include calamari_ocr into python code
'''
import os
from glob import glob

from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import run as calamari_ocr

import config
from bin.plugin import textExtPluginModel

# TODO:
# - convert to gray scale images
# - fixed width input 42 px


# Add your files here
calamari_models = ['vitaflow/annotate_server/static/data/calamari_models/model_00037200.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00045600.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00114400.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00123200.ckpt',
                   'vitaflow/annotate_server/static/data/calamari_models/model_00131400.ckpt']

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


class ocrTesseract(textExtPluginModel):
    def inputs(self):
        self._inputs = glob(os.path.join(config.TEXT_IMAGES, '*'))[:10]
        print('collected files {}'.format(len(self._inputs)))

    def run(self):
        args.files = self._inputs
        calamari_ocr(args)


if __name__ == '__main__':
    calamari_ocr(args)
