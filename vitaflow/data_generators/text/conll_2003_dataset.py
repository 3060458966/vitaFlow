# Copyright 2018 The vitFlow Authors. All Rights Reserved.
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
CoNLL Dataset Class
"""

__all__ = ["CoNLL2003Dataset"]
__version__ = '0.0.1'
__author__ = 'Mageswaran Dhandapani <mageswaran1989@gmail.com>'

import os
import shutil

import pandas as pd
import gin
from tqdm import tqdm

# from vitaflow.utils.hyperparams import HParams
from vitaflow.internal import IPreprocessor
from vitaflow.internal.dataset_types.dataset_types import ICSVSeq2SeqType1
from vitaflow.iterators.text.vocabulary import SpecialTokens
from vitaflow.utils.print_helper import *
from vitaflow.utils.data_io import maybe_download
from vitaflow.utils import registry


@registry.register_problem
@gin.configurable
class CoNLL2003Dataset(IPreprocessor, ICSVSeq2SeqType1):
    """
    Downloads the data and converts the text file into CSV file for each sentence along with its tags.
        - CoNLL dataset : https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
        - GDrive : https://drive.google.com/open?id=1tdwxPJnnkyO-s1oHDETj89cfgLC2xp0c

    .. code-block:: text

        Sample Raw Data:

            EU NNP B-NP B-ORG
            rejects VBZ B-VP O
            German JJ B-NP B-MISC
            call NN I-NP O
            to TO B-VP O
            boycott VB I-VP O
            British JJ B-NP B-MISC
            lamb NN I-NP O
            . . O O

        Preprocessed Data:

            0,1,2,3
            EU,NNP,B-NP,B-ORG
            rejects,VBZ,B-VP,O
            German,JJ,B-NP,B-MISC
            call,NN,I-NP,O
            to,TO,B-VP,O
            boycott,VB,I-VP,O
            British,JJ,B-NP,B-MISC
            lamb,NN,I-NP,O
            .,.,O,O

        ~/vitaflow/CoNLL2003Dataset/
                raw_data/
                    train.txt
                    test.txt
                    val.txt
                preprocessed_data/
                    train/
                    val/
                    test/
    """

    def __init__(self,
                 experiment_name,
                 preprocessed_data_path,
                 experiment_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/"),
                 train_data_path="train",
                 validation_data_path="val",
                 test_data_path="test",
                 over_write=False,
                 minimum_num_words=5,
                 seperator="~"):
        """
        """

        self._experiment_name = experiment_name
        self._preprocessed_data_path = preprocessed_data_path
        self._experiment_root_directory = experiment_root_directory
        self._train_data_path = train_data_path
        self._validation_data_path = validation_data_path
        self._test_data_path = test_data_path
        self._over_write = over_write
        self._minimum_num_words = minimum_num_words
        self._seperator = seperator

        IPreprocessor.__init__(self,
                               experiment_name=experiment_name,
                               preprocessed_data_path=preprocessed_data_path,
                               experiment_root_directory=experiment_root_directory,
                               train_data_path=train_data_path,
                               validation_data_path=validation_data_path,
                               test_data_path=test_data_path)

        self._download_path = os.path.join(
            self._experiment_root_directory,
            self._experiment_name,
            "raw_data/"
        )
        self._link = "https://drive.google.com/open?id=1tdwxPJnnkyO-s1oHDETj89cfgLC2xp0c"

        maybe_download(urls=self._link,
                       path=self._download_path,
                       filenames="conll2003.zip",
                       extract=True)

        self._prepare_data()

    # @staticmethod
    # def default_hparams():
    #     """
    #     .. role:: python(code)
    #        :language: python
    #
    #     .. code-block:: python
    #
    #         {
    #             "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/" ,
    #             "experiment_name" : "CoNLL2003Dataset",
    #             "preprocessed_data_path" : "preprocessed_data",
    #             "train_data_path" : "train",
    #             "validation_data_path" : "val",
    #             "test_data_path" : "test",
    #             "minimum_num_words" : 5,
    #             "over_write" : False,
    #         }
    #
    #     Here:
    #
    #     "experiment_root_directory" : str
    #         Root directory where the data is downloaded or copied, also
    #         acts as the folder for any subsequent experimentation
    #
    #     "experiment_name" : str
    #         Name of the data set
    #
    #     "preprocessed_data_path" : str
    #         Folder path under `experiment_root_directory` where the preprocessed data
    #         should be stored
    #
    #     "train_data_path" : str
    #         Folder path under `experiment_root_directory` where the train data is stored
    #
    #     "validation_data_path" : str
    #         Folder path under `experiment_root_directory` where the validation data is stored
    #
    #     "test_data_path" : str
    #         Folder path under `experiment_root_directory` where the test data is stored
    #
    #     "minimum_num_words" : str
    #         Number of word to be considered for a sentence to be used down the flow
    #
    #     "over_write" : boolean
    #         Flag to over write the previous copy of the downloaded data
    #
    #
    #     :return: A dictionary of hyperparameters with default values
    #     """
    #     hparams = IPreprocessor.default_hparams()
    #
    #     update({
    #         "experiment_name": "CoNLL2003Dataset",
    #         "minimum_num_words": 5,
    #         "over_write": False,
    #     })
    #     return hparams

    def _create_target_directories(self):
        """
        To setup destination folders structure if not present.
        :return:
        """
        if os.path.exists(self.PREPROCESSED_DATA_OUT_DIR):
            if self._over_write:
                print_info("Deleting data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
                shutil.rmtree(self.PREPROCESSED_DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
                os.makedirs(self.PREPROCESSED_DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
        else:
            print_info("Creating data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
            os.makedirs(self.PREPROCESSED_DATA_OUT_DIR)

    def _conll_to_csv(self, txt_file_path, out_dir):
        """
        Converts CoNLL 2003 data set text files into CSV file for each
        example/statement.
        :param txt_file_path: Input text file path
        :param out_dir: Output directory to store CSV files
        :return: Creates files in the specified train/val/test paths
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            print_info("Found previous version at {}".format(out_dir))
            return

        print_info("Writing data to {}...".format(out_dir))

        # Read the text file
        df = pd.read_csv(txt_file_path,
                         sep=" ",
                         skip_blank_lines=False,
                         header=None).fillna(SpecialTokens.UNK_WORD)

        # Filter out the DOCSTART lines
        df = df[~df[0].str.contains("DOCSTART")]

        current_file = []

        for i in tqdm(range(len(df))):
            row = df.values[i]
            if row[0] != SpecialTokens.UNK_WORD:
                current_file.append(row)
            else:
                # Consider dumping files with size 2
                if len(current_file) > self._minimum_num_words:
                    current_file = pd.DataFrame(current_file)
                    current_file.to_csv(out_dir + "/{}.csv".format(i), index=False, sep=self._seperator)
                    current_file = []

    def _prepare_data(self):
        """
        Prepares the data for training 
        :return: 
        """
        print_info("Preprocessing the train data...")
        self._conll_to_csv(os.path.join(self._download_path, "train.txt"),
                           self.TRAIN_OUT_PATH)

        print_info("Preprocessing the test data...")
        self._conll_to_csv(os.path.join(self._download_path, "test.txt"),
                           self.TEST_OUT_PATH)

        print_info("Preprocessing the validation data...")
        self._conll_to_csv(os.path.join(self._download_path, "val.txt"),
                           self.VAL_OUT_PATH)
