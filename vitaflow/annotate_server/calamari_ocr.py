'''
Demo sample example of how to include calamari_ocr into python code
'''
import os
from glob import glob

from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import calamari_ocr

import config
from bin.plugin import textExtraction


# TODO:
# - convert to gray scale images
# - fixed width input 42 px


class args:
    batch_size = 1
    checkpoint = ['/Users/sampathm/model_00131400.ckpt']  # Add your files here
    dataset = DataSetType.FILE
    extended_prediction_data = False
    extended_prediction_data_format = 'json'
    files = ['/Users/sampathm/10005.jpg',
             '/Users/sampathm/10006.jpg',
             '/Users/sampathm/10504.jpg',
             '/Users/sampathm/new2.png',
             '/Users/sampathm/X51008123447.jpg']  # Add your files here
    no_progress_bars = False
    output_dir = None
    pagexml_text_index = 1
    processes = 1
    text_files = None
    verbose = False
    voter = 'confidence_voter_default_ctc'


class ocrTesseract(textExtraction):
    def inputs(self):
        self._inputs = glob(os.path.join(config.TEXT_IMAGES, '*'))[:10]
        print('collected files {}'.format(len(self._inputs)))

    def run(self):
        args.files = self._inputs
        calamari_ocr(args)


if __name__ == '__main__':
    calamari_ocr(args)
