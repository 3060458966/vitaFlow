import warnings
import random
from overrides import overrides
import time
import threading

import numpy as np
import librosa
from numpy.lib import stride_tricks
import os


import tensorflow as tf
from tensorflow import TensorShape, Dimension
from tqdm import tqdm

from vitaflow.deprecated.playground import ShabdaWavPairFeature
from vitaflow.internal import HParams, IIteratorBase
from vitaflow.utils.print_helper import print_error, print_info, print_warn
# from vitaflow.core import HParams, IIteratorBase
# from vitaflow.helpers.print_helper import print_error, print_info, print_warn


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


class TEDLiumIterator(IIteratorBase, ShabdaWavPairFeature):
    def __init__(self, hparams=None, dataset=None):
        warnings.warn("This takes lot of time to process the files even with "
                      "TF data parallel support: examples.shabda.tedlium_parallel_iterator.TEDLiumIterator",
                      DeprecationWarning)

        IIteratorBase.__init__(self, hparams=hparams)
        ShabdaWavPairFeature.__init__(self)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())
        self._dataset = dataset
        print(dataset)

        self.TRAIN_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.VAL_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.TEST_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)

        if self._hparams.reinit_file_pair:
            self.TRAIN_WAV_PAIR = self.generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

        else:
            self.TRAIN_WAV_PAIR = self.read_pickle("train_wav_pair.p")
            if self.TRAIN_WAV_PAIR is None:
                self.TRAIN_WAV_PAIR = self.generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.read_pickle("val_wav_pair.p")
            if self.VAL_WAV_PAIR is None:
                self.VAL_WAV_PAIR = self.generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.read_pickle("test_wav_pair.p")
            if self.TEST_WAV_PAIR is None:
                self.TEST_WAV_PAIR = self.generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")


    @staticmethod
    def default_hparams():

        params = IIteratorBase.default_hparams()
        params.update( {
            "sampling_rate" : 16000,
            "frame_size" : 256,
            "neff" : 129,
            "min_amp" : 10000,
            "amp_fac" : 10000,
            "threshold" : 40,
            # prams for pre-whitening
            "global_mean" : 44,
            "global_std" : 15.5,
            "frames_per_sample" : 100,
            "reinit_file_pair" : False,
            "prefetch_size" : 32,
            "num_parallel_calls" : 8,
            "dummy_slicing_dim" : 1247
        }
        )
        return params

    @property

    def num_labels(self):
        return -1

    @property
    def num_train_samples(self):
        count = len(self.TRAIN_WAV_PAIR)
        return count

    @property
    def num_val_samples(self):
        return len(self.VAL_WAV_PAIR)

    @property
    def num_test_samples(self):
        return len(self.TEST_WAV_PAIR)

    def get_speaker_files(self, data_dir):
        """

        :param data_dir: dir containing the training data (root_dir + speaker_dir + wavfiles)
        :returns:  speaker_wav_files (dict) : {speaker : [files]}
        """
        # get dirs for each speaker
        speakers_dirs = [os.path.join(data_dir, speaker) for speaker in os.listdir(data_dir) \
                         if os.path.isdir(os.path.join(data_dir, speaker))]

        speaker_wav_files_dict = {}

        # get the files in each speakers dir
        for speaker_dir in speakers_dirs:
            speaker = speaker_dir.split("/")[-1]
            wav_files = [os.path.join(speaker_dir, file) for file in os.listdir(speaker_dir) if file.endswith("wav")]
            for wav_file in wav_files:
                if speaker not in speaker_wav_files_dict:
                    speaker_wav_files_dict[speaker] = []
                speaker_wav_files_dict[speaker].append(wav_file)
        return speaker_wav_files_dict

    def get_size(self, wav_file):
        st = os.stat(wav_file)
        return st.st_size

    def generate_match_dict(self, speaker_wav_files_dict):
        wav_file_pair = {}
        # generate match dict
        for speaker_1 in tqdm(speaker_wav_files_dict, desc="pair_dict"):
            for wav_file_1 in tqdm(speaker_wav_files_dict[speaker_1], desc="pair_dict_inner"):
                size_1 = self.get_size(wav_file_1)
                if size_1 < 1000 : #ignore invalid wav files
                    continue
                speaker_2 = random.choice(list(speaker_wav_files_dict))

                while (speaker_1 == speaker_2):
                    speaker_2 = random.choice(list(speaker_wav_files_dict))
                wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))

                wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                size_2 = self.get_size(wav_file_2)

                while size_2 < 1000:
                    wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                    wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                    size_2 = self.get_size(wav_file_2)

                wav_file_pair[wav_file_1] = wav_file_2

        return wav_file_pair


    def generate_features(self, wav_file_1, wav_file_2):

        try:
            start = time.time()
            speech_1, _ = librosa.core.load(wav_file_1, sr=self._hparams.sampling_rate)
            # amp factor between -3 dB - 3 dB
            fac = np.random.rand(1)[0] * 6 - 3
            speech_1 = 10. ** (fac / 20) * speech_1

            speech_2, _ = librosa.core.load(wav_file_2, sr=self._hparams.sampling_rate)
            fac = np.random.rand(1)[0] * 6 - 3
            speech_2 = 10. ** (fac / 20) * speech_2

            # mix
            length = min(len(speech_1), len(speech_2))
            speech_1 = speech_1[:length]
            speech_2 = speech_2[:length]
            speech_mix = speech_1 + speech_2

            # compute log spectrum for 1st speaker
            speech_1_features = np.abs(stft(speech_1, self._hparams.frame_size)[:, :self._hparams.neff])
            speech_1_features = np.maximum(speech_1_features, np.max(speech_1_features) / self._hparams.min_amp)
            speech_1_features = 20. * np.log10(speech_1_features * self._hparams.amp_fac)

            # same for the 2nd speaker
            speech_2_features = np.abs(stft(speech_2, self._hparams.frame_size)[:, :self._hparams.neff])
            speech_2_features = np.maximum(speech_2_features, np.max(speech_2_features) / self._hparams.min_amp)
            speech_2_features = 20. * np.log10(speech_2_features * self._hparams.amp_fac)

            # same for the mixture
            speech_mix_spec0 = stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
            speech_mix_features = np.abs(speech_mix_spec0)

            # speech_phase = speech_mix_spec0 / speech_mix_spec
            speech_mix_features = np.maximum(speech_mix_features, np.max(speech_mix_features) / self._hparams.min_amp)
            speech_mix_features = 20. * np.log10(speech_mix_features * self._hparams.amp_fac)
            max_mag = np.max(speech_mix_features)

            # if np.isnan(max_mag):
            # import ipdb; ipdb.set_trace()
            speech_VAD = (speech_mix_features > (max_mag - self._hparams.threshold)).astype(int)

            speech_mix_features = (speech_mix_features - self._hparams.global_mean) / self._hparams.global_std

            #The ideal binary mask gives ownership of a time-frequency bin to the source whose magnitude is
            # maximum among all sources in that bin.
            # The mask values were assigned with 1 for active and 0 otherwise (binary),
            # making Y x Y^T as the ideal affinity matrix for the mixture.
            Y = np.array([speech_1_features > speech_2_features, speech_1_features < speech_2_features]).astype('bool')
            Y = np.transpose(Y, [1, 2, 0]).astype('bool')

            # speech_mix_features = speech_mix_features[0:self._hparams.dummy_slicing_dim, :]
            # speech_VAD = speech_VAD[0:self._hparams.dummy_slicing_dim, :]
            # Y = Y[0:self._hparams.dummy_slicing_dim, :, :]

            # print_info("{} vs {}".format(wav_file_1, wav_file_2))
            end = time.time()


            print_info("Thread name: {} : took {}".format(threading.currentThread().getName(), end -start))

            if speech_mix_features.shape[0] != 1247 or speech_VAD.shape[0] != 1247 or Y.shape[0] != 1247:
                raise Exception("Found files with improper duration/data")

            return speech_mix_features.astype('float32'), speech_VAD.astype('bool'), Y.astype('bool')
        except Exception as e:
            print_warn(e)
            print_error("{} vs {}".format(wav_file_1, wav_file_2))
            return np.random.random((self._hparams.dummy_slicing_dim,129)).astype('float32'), \
                   np.empty((self._hparams.dummy_slicing_dim,129), dtype="bool"), \
                   np.empty((self._hparams.dummy_slicing_dim,129, 2), dtype="bool")

    def _user_resize_func(self, sample, vad, label):
        """
        Function that sets up the sizes of the tensor, after execution of `tf.py_func` call
        :param data:
        :param label:
        :return:
        """

        sample = tf.reshape(sample, shape=TensorShape([Dimension(self._hparams.dummy_slicing_dim),
                                                       Dimension(self._hparams.neff)]))
        vad = tf.reshape(vad, shape=TensorShape([Dimension(self._hparams.dummy_slicing_dim),
                                                 Dimension(self._hparams.neff)]))
        label = tf.reshape(label, shape=TensorShape([Dimension(self._hparams.dummy_slicing_dim),
                                                     Dimension(self._hparams.neff),
                                                     Dimension(2)]))
        return ({self.FEATURE_1_NAME: sample,
                 self.FEATURE_2_NAME: vad}, label)

    def _get_predict_samples(self, data_dir):
        speech_mix, _ = librosa.load(data_dir, self._hparams.sampling_rate)
        # fix the issue at the begining
        speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)
        speech_mix_spec0 = stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / speech_mix_spec
        speech_mix_spec = np.maximum(
            speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)
        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
        speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std
        len_spec = speech_mix_spec.shape[0]
        k = 0
        self.ind = 0
        self.samples = []
        # feed the transformed data into a sample list
        while (k + self._hparams.frames_per_sample < len_spec):
            phase = speech_phase[k: k + self._hparams.frames_per_sample, :]
            sample_mix = speech_mix_spec[k:k + self._hparams.frames_per_sample, :]
            VAD = speech_VAD[k:k + self._hparams.frames_per_sample, :]
            sample_dict = {'Sample': sample_mix,
                           'VAD': VAD,
                           'Phase': phase}
            self.samples.append(sample_dict)
            k = k + self._hparams.frames_per_sample
            # import ipdb; ipdb.set_trace()
        n_left = self._hparams.frames_per_sample - len_spec + k
        print(n_left)
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, self._hparams.neff])))
        sample_mix = np.concatenate(
            (speech_mix_spec[k:, :], np.zeros([n_left, self._hparams.neff])))
        VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, self._hparams.neff])))

        sample_dict = {'Sample': sample_mix,
                       'VAD': VAD,
                       'Phase': phase}
        self.samples.append(sample_dict)
        self.tot_samp = len(self.samples)
        begin = self.ind
        if begin >= self.tot_samp:
            return None
        self.ind += 1
        return (self.samples[begin], np.zeros(1))


    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """

        dataset = tf.data.Dataset.from_tensor_slices((list(self.TRAIN_WAV_PAIR.keys()),
                                                      list(self.TRAIN_WAV_PAIR.values())))
        dataset = dataset.map(
            lambda wav_file_1, wav_file_2: tuple(tf.py_func(
                self.generate_features, [wav_file_1, wav_file_2], (tf.float32, tf.bool, tf.bool))),
            num_parallel_calls=self._hparams.num_parallel_calls)

        dataset = dataset.map(self._user_resize_func, num_parallel_calls=self._hparams.num_parallel_calls)
        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """

        dataset = tf.data.Dataset.from_tensor_slices((list(self.VAL_WAV_PAIR.keys()),
                                                      list(self.VAL_WAV_PAIR.values())))
        dataset = dataset.map(
            lambda wav_file_1, wav_file_2: tuple(tf.py_func(
                self.generate_features, [wav_file_1, wav_file_2], (tf.float32, tf.bool, tf.bool))),
            num_parallel_calls=self._hparams.num_parallel_calls)
        dataset = dataset.map(self._user_resize_func, num_parallel_calls=self._hparams.num_parallel_calls)

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "val_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        dataset = tf.data.Dataset.from_generator(self._yield_test_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(self._hparams.frames_per_sample),
                                                                             Dimension(self._hparams.neff)]),
                                                                TensorShape([Dimension(self._hparams.frames_per_sample),
                                                                             Dimension(self._hparams.neff)]),
                                                                TensorShape([Dimension(self._hparams.frames_per_sample),
                                                                             Dimension(self._hparams.neff),
                                                                             Dimension(2)])))

        dataset = dataset.map(self.feature_map_func, num_parallel_calls=self._hparams.num_parallel_calls)

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "test_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def _get_predict_single_input_function(self, data):
        print("data: ", data)
        # dataset = tf.data.Dataset.from_tensor_slices(list(data))
        dataset1 = self._get_predict_samples(data)
        dataset = tf.data.Dataset.from_tensor_slices(dataset1)

        # dataset = dataset.map(self._user_resize_func, num_parallel_calls=self._hparams.num_parallel_calls)
        # dataset = ({self.FEATURE_1_NAME: dataset[0]['Sample'],
        #         self.FEATURE_2_NAME: dataset[0]['VAD']})
        dataset = dataset.batch(batch_size=100, drop_remainder=True)
        dataset = dataset.prefetch(100)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "val_data_cache"))

        print(dataset)
        return dataset

    @overrides
    def predict_on_instance(self, data):
        print(data)
        for i in data:
            print(i)
