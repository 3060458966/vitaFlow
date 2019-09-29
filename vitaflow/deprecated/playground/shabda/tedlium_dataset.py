# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
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

import os
import random
import numpy as np

from tqdm import tqdm
from pyspark.sql import SparkSession

from vitaflow.deprecated.playground import to_clips, _generate_tf_Records
from vitaflow.utils.hyperparams import HParams
from vitaflow.internal import DatasetInterface
from vitaflow.utils.print_helper import print_info
# from vitaflow.core import IPreprocessor, HParams
# from vitaflow.helpers.print_helper import print_info

class TEDLiumDataset(DatasetInterface):
    """
    Download the dataset from https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release1.tar.gz
    And unzip as shown below

    .. code-block:: text

        ~/vitaflow/                     #this can be anywhere on your system or on cloud ()#TODO
            TEDLiumDataset/
                raw_data/
                    train/
                        sph/
                            speaker_1.sph
                            speaker_2.sph ...
                    dev/
                        sph/
                            speaker_1.sph ...
                    test/
                        sph/
                            speaker_1.sph ...

                    processed_data/ #created as part of this class output
                        train/
                            clips/
                               speaker_1/
                                    file_1.wav
                                    file_2.wav ...
                                speaker_2/
                                    file_1.wav
                                    file_2.wav ...
                            tfrecords/
                                0.tfrecord
                                1.tfrecord ...

                        dev/
                            clips/
                                speaker_1/
                                    file_1.wav ...
                                speaker_2/
                                    file_1.wav ...
                            tfrecords/
                                0.tfrecord
                                1.tfrecord ...
                        test/
                            clips/
                                speaker_1/
                                    file_1.wav ...
                            tfrecords/
                                0.tfrecord
                                1.tfrecord ...
    """
    def __init__(self, hparams=None):
        DatasetInterface.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())

        self._spark_master = self._hparams.spark_master
        self._num_clips = self._hparams.num_clips
        self._duration = self._hparams.duration

        self._is_spark_initialized = False

        self.preprocess_prepare()

    def _initialize_spark(self):
        if not self._is_spark_initialized:
            self.spark = SparkSession.builder. \
            master(self._spark_master). \
            config("spark.executor.memory", "1g").\
            appName("shabda").getOrCreate()

            self.sc = self.spark.sparkContext
            self._is_spark_initialized = True

    def _prepare_wav_pairs(self):
        # get dict of {speaker : [files], ...}

        self.TRAIN_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=os.path.join(self.TRAIN_OUT_PATH, "clips"))
        self.VAL_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=os.path.join(self.VAL_OUT_PATH, "clips"))
        self.TEST_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=os.path.join(self.TEST_OUT_PATH, "clips"))

        if self._hparams.reinit_file_pair:
            # Generate speaker pair
            self.TRAIN_WAV_PAIR = self._generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self._generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self._generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

        else:
            self.TRAIN_WAV_PAIR = self.read_pickle("train_wav_pair.p")
            if self.TRAIN_WAV_PAIR is None:
                self.TRAIN_WAV_PAIR = self._generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.read_pickle("val_wav_pair.p")
            if self.VAL_WAV_PAIR is None:
                self.VAL_WAV_PAIR = self._generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.read_pickle("test_wav_pair.p")
            if self.TEST_WAV_PAIR is None:
                self.TEST_WAV_PAIR = self._generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "experiment_name",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test"
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name for the current experiment

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored

        #TODO add doc

        :return: A dictionary of hyperparameters with default values
        """
        hparams = DatasetInterface.default_hparams()

        hparams.update({
            "experiment_name": "TEDLiumDataset",
            "spark_master" : "local[4]",
            "num_clips" : 25,
            "sampling_rate" : 16000,
            "duration": 30,
            "frame_size": 256,
            "neff": 129,
            "min_amp": 10000,
            "amp_fac": 10000,
            "threshold": 40,
            # prams for pre-whitening
            "global_mean": 44,
            "global_std": 15.5,
            "frames_per_sample": 100,
            "reinit_file_pair" : False,
        })

        return hparams

    def _get_speaker_files(self, data_dir): #TODO S3 support
        """

        :param data_dir: dir containing the training data (root_dir + speaker_dir + wavfiles)
        :returns:  speaker_wav_files (dict) : {speaker : [files]}
        """
        # get dirs for each speaker
        speakers_dirs = [os.path.join(data_dir, speaker) for speaker in os.listdir(data_dir) \
                         if os.path.isdir(os.path.join(data_dir, speaker))]
        print_info(speakers_dirs)

        speaker_wav_files_dict = {}

        # get the files in each speakers dir
        # TODO: Convert below to dict-comprehension using collections.defaultdict
        for speaker_dir in speakers_dirs:
            speaker = speaker_dir.split("/")[-1]
            wav_files = [os.path.join(speaker_dir, file) for file in os.listdir(speaker_dir) if file.endswith("wav")]
            for wav_file in wav_files:
                if speaker not in speaker_wav_files_dict:
                    speaker_wav_files_dict[speaker] = []
                speaker_wav_files_dict[speaker].append(wav_file)

        if len(speaker_wav_files_dict) == 0:
            raise RuntimeError("shabda: No files are not under directory .... {}".format(data_dir))

        return speaker_wav_files_dict

    def _get_size(self, wav_file):
        """
        Finds the size of the given file
        :param wav_file: Wav file
        :return: int Size in bytes
        """
        st = os.stat(wav_file)
        return st.st_size

    def _generate_match_dict(self, speaker_wav_files_dict):
        """
        Generates pair of files from given dict of speakers and their speeches
        :param speaker_wav_files_dict: {speaker : [files]}
        :return: dict of files to files
        """
        wav_file_pair = {}
        # generate match dict
        for speaker_1 in tqdm(speaker_wav_files_dict, desc="pair_dict"):
            for wav_file_1 in speaker_wav_files_dict[speaker_1]:
                size_1 = self._get_size(wav_file_1)
                if size_1 < 100: #ignore files < 100 bytes
                    continue
                speaker_2 = random.choice(list(speaker_wav_files_dict))

                while (speaker_1 == speaker_2): #make sure we are not using the same speakers for mixture
                    speaker_2 = random.choice(list(speaker_wav_files_dict))

                wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                size_2 = self._get_size(wav_file_2)

                while size_2 < 100: #ignore files < 100 bytes
                    wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                    wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                    size_2 = self._get_size(wav_file_2)

                wav_file_pair[wav_file_1] = wav_file_2

        return wav_file_pair

    def _extract_clips(self, in_path, out_path):
        wav_files_n_data = self.sc.binaryFiles(in_path) # [(file_path, binary_data), ...]
        wav_files_n_data.foreach(lambda file_path_n_data :
                                 to_clips(file_path=file_path_n_data[0],
                                          data=file_path_n_data[1],
                                          num_clips=25,#self._num_clips,
                                          duration=30,#self._duration,
                                          output_dir=out_path))

    def _generate_mix_speeches(self, wav_pairs, out_dir):
        sampling_rate = self._hparams.sampling_rate
        frame_size = self._hparams.frame_size
        amp_fac = self._hparams.amp_fac
        neff = self._hparams.neff
        min_amp = self._hparams.min_amp
        threshold = self._hparams.threshold
        global_mean = self._hparams.global_mean
        global_std = self._hparams.global_std
        frames_per_sample = self._hparams.frames_per_sample

        speech_pair_rdd = self.sc.parallelize(wav_pairs.items()) # [(wav_1, wav_2), (wav_n, wav_m), ...]
        speech_pair_rdd = speech_pair_rdd.zipWithIndex() # [((wav_1, wav_2), 1), ... ((wav_n, wav_m), N) ...]

        # sdf
        speech_pair_rdd.foreach(lambda tuples : _generate_tf_Records(index=tuples[1],
                                                                     wav_file_1=tuples[0][0],
                                                                     wav_file_2=tuples[0][1],
                                                                     out_dir=out_dir,
                                                                     sampling_rate = sampling_rate,
                                                                     frame_size = frame_size,
                                                                     amp_fac = amp_fac,
                                                                     neff = neff,
                                                                     min_amp = min_amp,
                                                                     threshold = threshold,
                                                                     global_mean = global_mean,
                                                                     global_std = global_std,
                                                                     frames_per_sample = frames_per_sample))

    def preprocess_prepare(self):
        if not os.path.exists(os.path.join(self.TRAIN_OUT_PATH, "clips")):
            self._initialize_spark()
            self._extract_clips(os.path.join(self.TRAIN_IN_PATH, "sph"), os.path.join(self.TRAIN_OUT_PATH, "clips"))
        if not os.path.exists(os.path.join(self.VAL_OUT_PATH, "clips")):
            self._initialize_spark()
            self._extract_clips(os.path.join(self.VAL_IN_PATH, "sph"), os.path.join(self.VAL_OUT_PATH, "clips"))
        if not os.path.exists(os.path.join(self.TEST_OUT_PATH, "clips")):
            self._initialize_spark()
            self._extract_clips(os.path.join(self.TEST_IN_PATH, "sph"), os.path.join(self.TEST_OUT_PATH, "clips"))

        self._prepare_wav_pairs()

        if not os.path.exists(os.path.join(self.TRAIN_OUT_PATH, "tfrecords")):
            self._initialize_spark()
            print_info("Processing {} wav pairs, have a break...".format(len(self.TRAIN_WAV_PAIR)))
            self._generate_mix_speeches(self.TRAIN_WAV_PAIR, os.path.join(self.TRAIN_OUT_PATH, "tfrecords"))
        if not os.path.exists(os.path.join(self.VAL_OUT_PATH, "tfrecords")):
            self._initialize_spark()
            print_info("Processing {} wav pairs, have a break...".format(len(self.VAL_WAV_PAIR)))
            self._generate_mix_speeches(self.VAL_WAV_PAIR, os.path.join(self.VAL_OUT_PATH, "tfrecords"))
        if not os.path.exists(os.path.join(self.TEST_OUT_PATH, "tfrecords")):
            self._initialize_spark()
            print_info("Processing {} wav pairs, have a break...".format(len(self.TEST_WAV_PAIR)))
            self._generate_mix_speeches(self.TEST_WAV_PAIR, os.path.join(self.TEST_OUT_PATH, "tfrecords"))


