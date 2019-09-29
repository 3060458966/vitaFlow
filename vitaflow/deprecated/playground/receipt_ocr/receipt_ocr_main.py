# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main file that puts different pieces together for the issue #14
"""
# from memory_profiler import profile
import importlib
import os
import sys

import tensorflow as tf

sys.path.append(os.path.abspath('.'))

from examples.receipt_ocr.image_cropping import ImageCropping
from examples.receipt_ocr.image_segmentation import image_annotations
from examples.receipt_ocr.text_post_processing import TextPostProcessor
from vitaflow.utils.ocr.tesseract import TesseractOCR

flags = tf.flags
flags.DEFINE_string("config_python_file", "config_python_file", "The config to use.")


FLAGS = flags.FLAGS
config = FLAGS.config_python_file.replace("/", ".")
config = config.replace(".py", "")
config = importlib.import_module(config)

def main():

    annotated_data = image_annotations(path_to_tensorflow_model=config.path_to_tensorflow_model,
                      category_index=config.category_index,
                      images_src=config.images_src,
                      images_dest=config.images_dest)

    for each in annotated_data:
        t = ImageCropping(each['image_loc'], each['dest'])
        try:
            t.multiple_crop_and_save(each['coords'], each['tags'])
        except:
            print('Issues with parsing - {}'.format(each['image_loc']))

    ocr = TesseractOCR(image_dir=config.images_dest,
                       text_out_dir=config.text_out_dir)

    ocr.parallel_convert()

    text_post_processing = TextPostProcessor(text_out_dir=config.text_out_dir, receipt_text_dir=config.receipt_text_dir)
    text_post_processing.process()

if __name__ =="__main__":
    main()








