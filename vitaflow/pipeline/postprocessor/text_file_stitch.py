# -*- coding: utf-8 -*-
# coding=utf-8
from __future__ import unicode_literals

"""
To run
    `PYTHONIOENCODING=utf-8 python3`

"""
import concurrent.futures
import os

import glob
# import config
from vitaflow.pipeline.interfaces.plugin import TextCombiner
from vitaflow.pipeline.postprocessor.ocr_tesseract import TessaractOcrPlugin
from vitaflow.pipeline.postprocessor.ocr_calamari import CalamariOcrPlugin
from tqdm import tqdm
from vitaflow import demo_config

os.environ['OMP_THREAD_LIMIT'] = '1'

class TextFile(TextCombiner):

    def _handle_file(self, in_file_path, out_file_path):

        print(in_file_path)
        print(out_file_path)

        list_of_in_txt_files = in_file_path

        for each_in_text_prediction in tqdm(list_of_in_txt_files):
            if os.path.isfile(out_file_path):
                #  read the file and append values
                with open(out_file_path, "a") as fd:
                    fd.write(open(each_in_text_prediction, "r").read())
                    fd.write("\n")
            else:
                # create a new file with headers
                with open(out_file_path, "w") as fd:
                    fd.write(open(each_in_text_prediction, "r").read())
                    fd.write("\n")

    def _handle_files(self, source_dir, destination_dir):
        """Plugin module should implement this to handle all the files in the given directory"""

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        predicted_outputs = {}

        tesseract = TessaractOcrPlugin.__name__
        calamari = CalamariOcrPlugin.__name__

        for each_dir in os.listdir(source_dir):
            in_files = self.get_all_input_files(source_dir=os.path.join(source_dir, each_dir),
                                                input_files_types=[".txt"])
            #sort based on file number
            in_files = sorted(in_files,
                   key=lambda x: int(os.path.splitext(os.path.basename(x).split(os.extsep)[0])[0]))

            predicted_outputs[tesseract] = [file for file in in_files if tesseract in file]
            # TODO use calamari args; calamari latest version support renaming the output file names
            predicted_outputs[calamari] = [file for file in in_files if tesseract not in file]

            self._handle_file(in_file_path=predicted_outputs[tesseract],
                              out_file_path=os.path.join(destination_dir, each_dir) + "_" + tesseract + ".txt")

            self._handle_file(in_file_path=predicted_outputs[calamari],
                              out_file_path=os.path.join(destination_dir, each_dir) + "_" + calamari + ".txt")


if __name__ == '__main__':
    tt = TextFile()
    print('--' * 55)
    tt.process_files(source_dir=demo_config.TEXT_OCR_DATA_DIR, destination_dir=demo_config.TEXT_OUT_DIR)
