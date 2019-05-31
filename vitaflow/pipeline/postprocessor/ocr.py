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

import config
from vitaflow.pipeline.interfaces.plugin import TextExtImagePluginModel
from vitaflow.pipeline.interfaces.utils import trim_file_ext

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


def main(source_file, destination_file=None):
    img = cv2.imread(source_file)
    out_path = trim_file_ext(source_file) + '.tesseract.txt'
    text = pytesseract.image_to_string(img, lang='eng', config=config.TESSERACT_CONFIG)
    with open(out_path, "w") as fd:
        fd.write(string_parser(text))
    return out_path


def main_parallel(image_list):
    completed_jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for img_path, out_file in zip(image_list, executor.map(main, image_list)):
            completed_jobs.append(
                (img_path.split("\\")[-1], ',', out_file, ', processed')
            )


class OcrTessaract(TextExtImagePluginModel):
    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.TEXT_IMAGES
        self.destination_folder = config.TEXT_IMAGES
        # Transformation function for converting source_image to destination_image
        self.operator_func = main
        self.parallel_operator_func = main_parallel


if __name__ == '__main__':
    tt = OcrTessaract()
    tt.plugin_inputs()
    print('--' * 55)
    tt.bulk_run()
