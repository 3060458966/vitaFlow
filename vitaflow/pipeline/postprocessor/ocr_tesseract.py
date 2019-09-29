# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import unicode_literals

"""
To run
    `PYTHONIOENCODING=utf-8 python3`

"""
import sys
import os
sys.path.append(os.getcwd())
import fire
import unicodedata

import cv2
import pytesseract

# import config
from vitaflow.pipeline.interfaces.plugin import OCRModuleInterface

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


class TessaractOcrModule(OCRModuleInterface):
    def __init__(self,
                 file_postfix=None,
                 num_workers=4,
                 tesseract_config='--oem 1 --psm 3 -l eng'):# -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -c preserve_interword_spaces=1'):
        # --psm 1 - -oem 1 - -dpi 300 tsv
        OCRModuleInterface.__init__(self, num_workers=num_workers, file_postfix=file_postfix)
        self._tesseract_config = tesseract_config

    def _handle_file(self, in_file_path, out_file_path):
        try:
            img = cv2.imread(in_file_path)
            text = pytesseract.image_to_string(img, lang='eng', config=self._tesseract_config)
            with open(out_file_path, "w") as fd:
                fd.write(string_parser(text))
            return out_file_path
        except:
            print("Failed : {} ----> {}".format(in_file_path, out_file_path))


def run(source_directory,
        destination_dir):
    """
    Utility to run Tesseract-OCR on cropped text images
    :param source_directory: Directory which has list of folders each folder containing cropped images of identified text regions
    :param destination_dir: Directory to store the extracted text from cropped images preserving the folder structure
    :return:
    """

    tt = TessaractOcrModule(num_workers=4)
    print('--' * 55)
    tt.process_files(source_dir=source_directory,
                     destination_dir=destination_dir)

if __name__ == '__main__':
    fire.Fire(run)
