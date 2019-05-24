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
import config
from bin.plugin import StitchTextExtPluginModel
from tqdm import tqdm

os.environ['OMP_THREAD_LIMIT'] = '1'


def main(source_file):
    source_file_folder = os.path.dirname(source_file)
    ext = os.path.splitext(os.path.basename(source_file).split(os.extsep)[1])[0]
    destination_file = os.path.join(source_file_folder, "output.{}.txt".format(ext))
    if os.path.isfile(destination_file):
        #  read the file and append values
        with open(destination_file, "a") as fd:
            fd.write(open(source_file, "r").read())
            fd.write("\n")
    else:
        # create a new file with headers
        with open(destination_file, "w") as fd:
            fd.write(open(source_file, "r").read())
            fd.write("\n")
    return destination_file


def main_parallel(text_list):
    completed_jobs = []
    # filter out the extensions based on the folders
    folders = glob.glob(text_list + os.sep + "*")

    # create a folder in output directory with name as input folder
    for folder_path in tqdm(folders, desc="text_post_processing"):
        for ext in config.OCR_TEXT_EXTS:
            files = sorted(glob.glob(folder_path + "/*{}".format(ext)),
                           key=lambda x: int(os.path.splitext(os.path.basename(x).split(os.extsep)[0])[0]))
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                for text_file_path, out_file in zip(files, executor.map(main, files)):
                    completed_jobs.append(
                        (text_file_path.split("\\")[-1], ',', out_file, ', processed')
                    )


class TextFileStitch(StitchTextExtPluginModel):
    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.TEXT_IMAGES
        self.destination_folder = config.TEXT_DIR
        # Transformation function for converting source_image to destination_image
        self.operator_func = main
        self.parallel_operator_func = main_parallel


if __name__ == '__main__':
    tt = TextFileStitch()
    tt.plugin_inputs()
    print('--' * 55)
    tt.bulk_run()
