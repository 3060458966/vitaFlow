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
import glob
import os
import pandas as pd

from tqdm import tqdm

class TextPostProcessor:

    def __init__(self,
                 text_out_dir,
                 receipt_text_dir):
        self._text_out_dir = text_out_dir
        self._receipt_text_dir = receipt_text_dir

        if not os.path.exists(receipt_text_dir):
            os.makedirs(receipt_text_dir)

    def process(self):
        folders = glob.glob(self._text_out_dir + os.sep + "*")


        for folder_path in tqdm(folders, desc="text_post_processing"):
            bag  = []
            file_name = folder_path.split("/")[-1]
            text_files = glob.glob(folder_path + os.sep + "*.txt")
            info_dict = {}

            for text_file in text_files:
                file_name = text_file.split("/")[-2]
                tag_name = text_file.split("/")[-1].split(".")[0]
                bag.append((tag_name, open(text_file, "r").read().replace(",", "")))

            df = pd.DataFrame(bag, columns=["Tag", "Text"])
            df.to_csv(os.path.join(self._receipt_text_dir, file_name + ".csv"))
            print(df)
