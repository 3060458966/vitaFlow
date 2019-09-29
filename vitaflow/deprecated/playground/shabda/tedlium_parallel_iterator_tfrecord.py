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

import glob
import os


import itertools

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import librosa

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from vitaflow.deprecated.playground import ShabdaWavPairFeature
from vitaflow.deprecated.playground import _stft
from vitaflow.utils.hyperparams import HParams
from vitaflow.internal import IIteratorBase
from vitaflow.utils.print_helper import print_info
from vitaflow.utils.print_helper import print_error
# from vitaflow.core import HParams, IIteratorBase
# from vitaflow.helpers.print_helper import print_info
# from vitaflow.helpers.print_helper import print_error


class TEDLiumIterator(IIteratorBase, ShabdaWavPairFeature):
    def __init__(self, hparams=None, dataset=None):
        """
        Uses the TEDLium preprocessed (speaker wise folders) data and generates the mixture signals
        of two speakers

        The features are as follows:
            - spectral features in log scale of two speaker voices. shape: [frames_per_sample, neff]
            - Voice activity detection array (0's/1's). shape: [frames_per_sample, neff]
            - Active Speaker array as labels. shape: [frames_per_sample, neff, 2]

        :param hparams:
        :param dataset:
        """
        IIteratorBase.__init__(self, hparams=hparams)
        ShabdaWavPairFeature.__init__(self)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())
        self._dataset = dataset



    @property
    def num_labels(self):
        """
        No labels associated with this data iterator, hence None
        :return: None
        """
        return None

    @property
    def num_train_samples(self):
        return len(self._dataset.TRAIN_WAV_PAIR)

    @property
    def num_val_samples(self):
        return len(self._dataset.VAL_WAV_PAIR)

    @property
    def num_test_samples(self):
        return len(self._dataset.TEST_WAV_PAIR)

    @staticmethod
    def default_hparams():
        #TODO add doc
        params = IIteratorBase.default_hparams()
        params.update({
            "sampling_rate": 16000,
            "frame_size": 256,
            "neff": 129,
            "min_amp": 10000,
            "amp_fac": 10000,
            "threshold": 40,
            # prams for pre-whitening
            "global_mean": 44,
            "global_std": 15.5,
            "frames_per_sample": 100,
            "reinit_file_pair": False,
            "prefetch_size": 32,
            "num_threads" : 4
        }
        )
        return params

    def decode(self, serialized_example):
        # 1. define a parser
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'speech_mix': tf.FixedLenFeature([self._hparams.frames_per_sample * self._hparams.neff], tf.float32),
                'VAD': tf.FixedLenFeature([self._hparams.frames_per_sample * self._hparams.neff], tf.int64),
                'Y': tf.FixedLenFeature([2*self._hparams.frames_per_sample * self._hparams.neff], tf.int64),
                'FPS': tf.FixedLenFeature([], tf.int64),
                'NEFF': tf.FixedLenFeature([], tf.int64),
            })
        FPS = tf.cast(features['FPS'], tf.int32)
        NEFF = tf.cast(features['NEFF'], tf.int32)
        sample_mix = tf.reshape(tf.cast(features['speech_mix'], tf.float32), shape=[self._hparams.frames_per_sample, self._hparams.neff])
        VAD = tf.reshape(tf.cast(features['VAD'], tf.float32), shape=[self._hparams.frames_per_sample, self._hparams.neff])
        Y = tf.reshape(tf.cast(features['Y'], tf.float32), shape=[self._hparams.frames_per_sample, self._hparams.neff, 2])

        return {self.FEATURE_1_NAME: sample_mix, self.FEATURE_2_NAME: VAD}, Y

    # 
    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: dataset
        """
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._dataset.TRAIN_OUT_PATH, "tfrecords/*.tfrecord")),
                                          num_parallel_reads=self._hparams.num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)

        return dataset

    #
    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._dataset.VAL_OUT_PATH, "tfrecords/*.tfrecord")),
                                          num_parallel_reads=self._hparams.num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset


    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._dataset.TEST_OUT_PATH, "tfrecords/*.tfrecord")),
                                          num_parallel_reads=self._hparams.num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset


    def _get_predict_samples(self, file_path):
        """

        :param file_path:
        :return:
        """
        speech_mix, _ = librosa.load(file_path, self._hparams.sampling_rate) # 10 seconds * 8000 SR = 80000
        print_info("speech_mix original shape: {} for a length of {} seconds with Sampling rate {}".format(speech_mix.shape,
                                                                                                           librosa.get_duration(y=speech_mix, sr=self._hparams.sampling_rate),
                                                                                                           self._hparams.sampling_rate))
        # replicate the speech signal thrice,1: original audio,  2:speaker one, 3: speaker two
        speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)
        print_info("speech_mix original shape: {} for a length of {} seconds with Sampling rate {}".format(speech_mix.shape,
                                                                                                           librosa.get_duration(y=speech_mix, sr=self._hparams.sampling_rate),
                                                                                                           self._hparams.sampling_rate))
        # feature enginerring like before
        speech_mix_spec0 = _stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / speech_mix_spec
        speech_mix_spec = np.maximum(speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)

        max_mag = np.max(speech_mix_spec)
        speech_voice_activity_detection = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
        speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std
        num_frames = speech_mix_spec.shape[0] # number of frames
        print_info("shape of speech_mix_spec : {}".format(speech_mix_spec.shape))
        print_info("num_frames in speech_mix_spec {}".format(num_frames)) # 3747

        current_frame = 0
        self.ind = 0
        self.freq_features = []
        self.voice_activity_detection_features = []
        self.phase_features = []

        # feed the transformed data into a sample list
        while (current_frame + self._hparams.frames_per_sample < num_frames):
            phase = speech_phase[current_frame: current_frame + self._hparams.frames_per_sample, :]
            sample_mix = speech_mix_spec[current_frame:current_frame + self._hparams.frames_per_sample, :]
            voice_activity_detection = speech_voice_activity_detection[current_frame:current_frame + self._hparams.frames_per_sample, :]
            self.freq_features.append(sample_mix.astype('float32'))
            self.voice_activity_detection_features.append(voice_activity_detection.astype('bool'))
            self.phase_features.append(phase)

            current_frame = current_frame + self._hparams.frames_per_sample
            # import ipdb; ipdb.set_trace()

        # Corner case for the frames that was left
        # 100 - (3747 3700) = 53
        remaining_frames = self._hparams.frames_per_sample - num_frames + current_frame

        # append zeros for missing frames from last position
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))
        sample_mix = np.concatenate((speech_mix_spec[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))
        voice_activity_detection = np.concatenate((speech_voice_activity_detection[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))

        self.freq_features.append(sample_mix.astype('float32'))
        self.voice_activity_detection_features.append(voice_activity_detection.astype('bool'))
        self.phase_features.append(phase)
        return np.asarray(self.freq_features),  self.voice_activity_detection_features, self.phase_features



    def predict_on_instance(self, executor, file_path):
        """
        Given a mixed audio file, it generates three audio files : mix recontructed, source1, source 2
        #TODO debug!!!
        :param executor:
        :param test_file_path:
        :return:
        """

        estimator = executor.estimator

        in_data_features, voice_activity_detection_data_features, phase_features = self._get_predict_samples(file_path=file_path)

        print_info("in_data_features original shape: {}".format(in_data_features.shape))

        voice_activity_detection_data_features = np.asarray(voice_activity_detection_data_features)

        num_samples_N = in_data_features.shape[0]
        hop_size = self._hparams.frame_size // 4 # 256 / 4 = 64

        # (38 * 100 -1) * 64 + 256 = 243392 for wav file of 10 seconds appended 3 three times i.e 10 * 8000 (Sampling rate) * = 240000
        out_audio1 = np.zeros([(num_samples_N * self._hparams.frames_per_sample - 1) * hop_size + self._hparams.frame_size])
        out_audio2 = np.zeros([(num_samples_N * self._hparams.frames_per_sample - 1) * hop_size + self._hparams.frame_size])
        mix = np.zeros([(num_samples_N *self._hparams.frames_per_sample - 1) * hop_size + self._hparams.frame_size])

        def get_dataset():
            dataset = tf.data.Dataset.from_tensor_slices((
                {self.FEATURE_1_NAME: in_data_features,
                 self.FEATURE_2_NAME: voice_activity_detection_data_features},
                np.ones_like(in_data_features)
            ))
            dataset = dataset.batch(batch_size=1)
            print_info(dataset.output_shapes)
            return dataset

        predict_fn = estimator.predict(input_fn=lambda: get_dataset())

        print_info("Shape of in data: {}".format(in_data_features.shape))
        print_info("Number of sample for given file: {}".format(num_samples_N))

        embeddings = []
        i = 0

        for predicted_value in predict_fn:
            # print("i = {}".format(i))
            """
            TODO:
            strange behaviour!
            
            1 wav file = N samples
            Eg: N = 600
            FramesPerSample=100, BatchSize = 1, NEFF = 129, EMD_K = 30
            
            For each sample the embeddings is of shape [batch_size * frames_per_sample, NEFF, embd_dim].
            For prediction batch size is made 1.
            Hence the embeddings colapse to [frames_per_sample, NEFF, embd_dim]
            1 sample predictions will have `frames_per_sample` outputs
            Eg: If input audio file has 75 frames, the prediction will have [7500, NEFF, embd_dim]
            """
            embeddings.append(predicted_value)
            i += 1

        # Number of embeddings predicted for given file: 3800 with shape (3800, 129, 40) [number of samples * frames_per_sample, NEFF, EMBD_DIM]
        print_info("Number of embeddings predicted for given file: {} with shape {}".format(len(embeddings), np.asarray(embeddings).shape))

        N_assign = 0
        # # for every chunk of frames of data
        for sample_i in tqdm(range(num_samples_N)): # num_samples = 38

            # expand the dimesion to be inline with TF batch size
            in_data_np = np.expand_dims(in_data_features[sample_i], axis=0)
            in_phase_np = np.expand_dims(phase_features[sample_i], axis=0)
            voice_activity_detection_data_np = np.expand_dims(voice_activity_detection_data_features[sample_i], axis=0)
            print_info("in_data_np : ")
            print_error(in_data_np.shape)
            """
            0*100 to (0+1)*100 = 0   to 100
            1*100 to (1+1)*100 = 100 to 200
            2*100 to (2+1)*100 = 200 to 300
            
            """
            embedding_np = np.asarray(embeddings[sample_i*self._hparams.frames_per_sample: (sample_i+1)*self._hparams.frames_per_sample])


            # ----------------------------------------------

            # embedding_ac = []
            # for i, j in itertools.product(range(self._hparams.frames_per_sample), range(self._hparams.neff)):
            #     if voice_activity_detection_data_np[0, i, j] == 1:
            #         embedding_ac.append(embedding_np[i, j, :])

            embedding_ac = [embedding_np[i, j, :]
                            for i, j in itertools.product(
                    range(self._hparams.frames_per_sample), range(self._hparams.neff))
                            if voice_activity_detection_data_np[0, i, j] == 1]
            print_error(np.array(embedding_ac).shape)
            if embedding_ac == []:
                break
            kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)

            # ----------------------------------------------

            mask = np.zeros([self._hparams.frames_per_sample, self._hparams.neff, 2])
            ind = 0

            # print_info("N_assign : {}".format(N_assign))

            center = kmean.cluster_centers_
            center = center * 0.7 + 0.3 * kmean.cluster_centers_
            # two speakers have appeared
            # print_info("Found 2 speakers ...")
            center_new = kmean.cluster_centers_
            cor = np.matmul(center_new[0, :], np.transpose(center))
            # rearrange their sequence if not consistant with previous
            # frames
            # print_info("Correlation : {}".format(cor))

            if(cor[1] > cor[0]):
                kmean.cluster_centers_ = np.array([kmean.cluster_centers_[1], kmean.cluster_centers_[0]])
                kmean.labels_ = (kmean.labels_ == 0).astype('int')

            kmean.labels_ = (~kmean.labels_).astype('int')


            # print_info("center : {}".format(center))
            # print_info("kmean.labels_ : {}".format(kmean.labels_))

            # ----------------------------------------------
            # print_error(kmean.labels_)
            # transform the clustering result and voice_activity_detection info. into masks
            for i in range(self._hparams.frames_per_sample):
                for j in range(self._hparams.neff):
                    if voice_activity_detection_data_np[0, i, j] == 1:
                        mask[i, j, kmean.labels_[ind]] = 1
                        ind += 1

            for i in range(self._hparams.frames_per_sample):
                # apply the mask and reconstruct the waveform
                tot_ind = sample_i * self._hparams.frames_per_sample + i
                # ipdb.set_trace()
                amp = in_data_np[0, i, :] * self._hparams.global_std + self._hparams.global_mean

                out_data1 = (mask[i, :, 0] * amp * voice_activity_detection_data_np[0, i, :])
                out_data2 = (mask[i, :, 1] * amp * voice_activity_detection_data_np[0, i, :])
                out_mix = amp

                out_data1_l = 10 ** (out_data1 / 20) / self._hparams.amp_fac
                out_data2_l = 10 ** (out_data2 / 20) / self._hparams.amp_fac
                out_mix_l = 10 ** (out_mix / 20) / self._hparams.amp_fac

                out_stft1 = out_data1_l * in_phase_np[0, i, :]
                out_stft2 = out_data2_l * in_phase_np[0, i, :]
                out_stft_mix = out_mix_l * in_phase_np[0, i, :]

                con_data1 = out_stft1[-2:0:-1].conjugate()
                con_data2 = out_stft2[-2:0:-1].conjugate()
                con_mix = out_stft_mix[-2:0:-1].conjugate()

                out1 = np.concatenate((out_stft1, con_data1))
                out2 = np.concatenate((out_stft2, con_data2))
                out_mix = np.concatenate((out_stft_mix, con_mix))

                frame_out1 = np.fft.ifft(out1).astype(np.float64)
                frame_out2 = np.fft.ifft(out2).astype(np.float64)
                frame_mix = np.fft.ifft(out_mix).astype(np.float64)

                start = tot_ind * hop_size
                out_audio1[start:(start + len(frame_out1))] += frame_out1 * 0.5016
                out_audio2[start:(start + len(frame_out2))] += frame_out2 * 0.5016
                mix[start:(start + len(frame_mix))] += frame_mix * 0.5016


        ## the audio has been padded 3 times in AudioReader
        ## restore the original audio
        len1 = len(out_audio1) // 3
        len2 = len(out_audio2) // 3
        source1 = out_audio1[len1:2*len1]
        source2 = out_audio2[len2:2*len2]
        mix = mix[len2:2*len2]

        # Length of source_1 81130 source_1 shape (81130,)
        # Length of source_1 81130 source_1 shape (81130,)

        print_info("Length of source_1 {} source_1 shape {}".format(len1, source1.shape))
        print_info("Length of source_1 {} source_1 shape {}".format(len2, source2.shape))
        print_info("Writing file {}".format(os.path.splitext(file_path)[0] + "_source1.wav"))
        print_info("Writing file {}".format(os.path.splitext(file_path)[0] + "_source2.wav"))

        librosa.output.write_wav(os.path.splitext(file_path)[0] + "_source1.wav", source1, self._hparams.sampling_rate)
        librosa.output.write_wav(os.path.splitext(file_path)[0] + "_source2.wav", source2, self._hparams.sampling_rate)
        librosa.output.write_wav(os.path.splitext(file_path)[0] + "_full.wav", mix, self._hparams.sampling_rate)
        return [(source1, self._hparams.sampling_rate), (source2, self._hparams.sampling_rate)]



    def visulaize(self, executor, file_path):
        """

        :param executor:
        :param test_file_path:
        :return:
        """

        estimator = executor.estimator

        in_data_features, voice_activity_detection_data_features, phase_features = self._get_predict_samples(file_path=file_path)

        in_data_features = np.asarray(in_data_features)
        voice_activity_detection_data_features = np.asarray(voice_activity_detection_data_features)

        N_frames = in_data_features.shape[0]
        hop_size = self._hparams.frame_size // 4

        def get_dataset():
            dataset = tf.data.Dataset.from_tensor_slices((
                {self.FEATURE_1_NAME: in_data_features,
                 self.FEATURE_2_NAME: voice_activity_detection_data_features},
                np.ones_like(in_data_features)
            ))
            dataset = dataset.batch(batch_size=1)
            print_info(dataset.output_shapes)
            return dataset

        predict_fn = estimator.predict(input_fn=lambda: get_dataset())

        print_info("Shape of in data: {}".format(in_data_features.shape))
        print_info("Number of frames for given file: {}".format(N_frames))

        embeddings = []
        i = 0

        for predicted_value in predict_fn:
            # print("i = {}".format(i))
            """
            TODO:
            strange behaviour!
            
            1 wav file = N samples
            Eg: N = 600
            FramesPerSample=100, BatchSize = 1, NEFF = 129, EMD_K = 30
            
            For each sample the embeddings is of shape [batch_size * frames_per_sample, NEFF, embd_dim].
            For prediction batch size is made 1.
            Hence the embeddings colapse to [frames_per_sample, NEFF, embd_dim]
            1 sample predictions will have `frames_per_sample` outputs
            Eg: If input audio file has 75 frames, the prediction will have [7500, NEFF, embd_dim]
            """
            embeddings.append(predicted_value)
            i += 1

        print_info("Number of embeddings predicted for given file: {}".format(len(embeddings)))
        print_error(np.asarray(embeddings).shape)

        N_assign = 0
        step = 0

        for frame_i in tqdm(range(N_frames)):

            # expand the dimesion to be inline with TF batch size
            in_data_np = np.expand_dims(in_data_features[frame_i], axis=0)
            in_phase_np = np.expand_dims(phase_features[frame_i], axis=0)
            voice_activity_detection_data_np = np.expand_dims(voice_activity_detection_data_features[frame_i], axis=0)
            embedding_np = np.asarray(embeddings[frame_i: frame_i+self._hparams.frames_per_sample])

            # ----------------------------------------------

            embedding_ac = []
            for i, j in itertools.product(range(self._hparams.frames_per_sample), range(self._hparams.neff)):
                if voice_activity_detection_data_np[0, i, j] == 1:
                    embedding_ac.append(embedding_np[i, j, :])

            kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)
            # visualization using 3 PCA
            pca_Data = PCA(n_components=3).fit_transform(embedding_ac)
            fig = plt.figure(1, figsize=(8, 6))
            ax = Axes3D(fig, elev=-150, azim=110)
            # ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
            #            c=kmean.labels_, cmap=plt.cm.Paired)
            ax.scatter(pca_Data[:, 0], pca_Data[:, 1], pca_Data[:, 2],
                       cmap=plt.cm.Paired)
            ax.set_title('Embedding visualization using the first 3 PCs')
            ax.set_xlabel('1st pc')
            ax.set_ylabel('2nd pc')
            ax.set_zlabel('3rd pc')
            if not os.path.exists("vis"):
                os.makedirs("vis")
            plt.savefig('vis/' + str(step) + 'pca.jpg')

            step += 1
