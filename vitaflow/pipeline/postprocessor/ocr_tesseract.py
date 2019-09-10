# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import unicode_literals

"""
To run
    `PYTHONIOENCODING=utf-8 python3`

"""
import concurrent.futures
import os
import unicodedata

import cv2
import pytesseract

# import config
from vitaflow.pipeline.interfaces.plugin import OCRPluginInterface
from vitaflow.pipeline.interfaces.utils import trim_file_ext
from vitaflow import demo_config

os.environ['OMP_THREAD_LIMIT'] = '1'


def string_parser(text):
    debug = False
    if debug:
        print('--' * 15)
        print(text)
        print('--' * 15)
    try:
        # TODO SAMPATH Check why this normalization was plugged in
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
        text = text.decode('utf-8')
        # text
    except:
        text = ""
    return str(text)


class TessaractOcrPlugin(OCRPluginInterface):
    def __init__(self,
                 num_workers=4,
                 tesseract_config='-oem -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1'):
        # --psm 1 - -oem 1 - -dpi 300 tsv
        OCRPluginInterface.__init__(self, num_workers=num_workers)
        self._tesseract_config = tesseract_config

    def _handle_file(self, in_file_path, out_file_path):
        img = cv2.imread(in_file_path)
        text = pytesseract.image_to_string(img, lang='eng', config=self._tesseract_config)
        with open(out_file_path, "w") as fd:
            fd.write(string_parser(text))
        return out_file_path


if __name__ == '__main__':
    tt = TessaractOcrPlugin(num_workers=4)
    print('--' * 55)
    tt.process_files(source_dir=demo_config.CROPPER_ROOT_DIR,
                     destination_dir=demo_config.TEXT_OCR_DATA_DIR)
