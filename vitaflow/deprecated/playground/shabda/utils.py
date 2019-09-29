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
import traceback
import io

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from numpy.lib import stride_tricks
from scipy.io import wavfile

from vitaflow.utils.print_helper import print_error


def to_clips(file_path, data, num_clips=128, duration=20, output_dir="/tmp/"):
    print(file_path)
    file_path = file_path.replace("file:", "")
    tmp = io.BytesIO(data)
    wav_data, sampling_rate = sf.read(tmp)

    speaker = file_path.split("/")[-1].split(".")[0]
    speaker_dir = os.path.join(output_dir, speaker)
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)

    y, _ = librosa.load(file_path, sr=sampling_rate)
    end_time = librosa.get_duration(y=y, sr=sampling_rate)
    for j in range(num_clips):
        wav_file = os.path.join(speaker_dir, str(j)) + ".wav"
        k = int(np.random.randint(0, end_time-duration, size=1))
        librosa.output.write_wav(wav_file,
                                 y[k*sampling_rate : (k+duration)*sampling_rate],
                                 sampling_rate)

    return None


def _get_speech_data(wav_file, sampling_rate): #TODO S3 support
    """

    :param wav_file:
    :param sampling_rate:
    :return:
    """
    # speech, _ = librosa.core.load(wav_file, sr=sampling_rate)
    sr, speech = wavfile.read(wav_file)
    # amplication factor between -3 (dB) to 3 (dB)
    fac = np.random.rand(1)[0] * 6 - 3
    # randomly choose the amplification factor and applyt he factor to get gan or loss of the speech
    # why so? randomly we are increasing/decreasing or amplification/damping the magnitude of the signals
    # dB = 20 * log_10(p2/p1)
    # fac / 20 = log_10(p2/p1)
    # 10 ^ (fac/20) = p2/p1
    #convert the `fac` to decibel and applyt it to the signal
    #fac_db = 10. ^ (fac / 20)
    speech = 10. ** (fac / 20) * speech
    return speech

#@profile
def _stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """
    Short time fourier transform of audio signal
    computing STFTs is to divide a longer time signal into shorter segments of equal length and then compute
    the Fourier transform separately on each shorter segment.
    This reveals the Fourier spectrum on each shorter segment.

    Reference:
    - https://en.wikipedia.org/wiki/Hann_function

    :param sig: 10 seconds 8000 SR wav file. 30 seconds 8000 SR wav file.
    :param frameSize: 256 default
    :param overlapFac:
    :param window:
    :return:
    """

    #Eg :  30 seconds 8000 SR wav file.

    win = window(frameSize)

    # 256  - (0.75 * 256) = 64
    hopSize = int(frameSize - np.floor(overlapFac * frameSize)) # number of slides
    samples = np.array(sig, dtype='float64') # 240000

    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1) # 3747
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize)) # 240256

    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win # (3747, 256)
    #discrete Fourier Transform
    return np.fft.rfft(frames)

#@profile
def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

#@profile
def _get_log_spectrum_features(speech, frame_size, neff, amp_fac, min_amp):
    """
    find short time fourier transform for the mixture and trim it to given NEFF
        -  https://www.cds.caltech.edu/~murray/wiki/Why_is_dB_defined_as_20log_10%3F
        -  http://www.sengpielaudio.com/calculator-FactorRatioLevelDecibel.htm
    :param speech:
    :param frame_size:
    :param neff:
    :param amp_fac:
    :param min_amp:
    :return:
    """
    speech_features = np.abs(_stft(speech, frame_size)[:, :neff])
    # say if we have 1256 max of speech signal array and min_amp as follows
    # 1256/1000 = maximum(speech, 1.256)
    # 1256/50 = maximum(speech, 25.12)
    # 1256/10 = maximum(speech, 125.6)
    # 1256/5  = maximum(speech, 251.2)
    # 1256/1  = maximum(speech, 1256)
    # np.max : array wise
    # np.maximum : element wise
    # scaling the speech features with respect to min_amp and its max value
    speech_features = np.maximum(speech_features, np.max(speech_features) / min_amp)
    speech_features = replaceZeroes(speech_features)
    speech_features = 20. * np.log10(speech_features * amp_fac)
    return speech_features

#@profile
def _get_speech_samples(speech_1_features,
                        speech_2_features,
                        frames_per_sample,
                        number_frames,
                        speech_mix_features,
                        speech_voice_activity_detection):
    """
    Processes one audio pair
    :param speech_1_features: Spectral audio features
    :param speech_2_features: Spectral audio features
    :param frames_per_sample:
    :param len_spec: Total number of frames (to say?) in the given signal
    :param speech_mix_features:
    :param speech_voice_activity_detection:
    :return: Returns an array of shape ((self._hparams.frames_per_sample, self._hparams.neff),(self._hparams.frames_per_sample, self._hparams.neff),(self._hparams.frames_per_sample, self._hparams.neff, 2))
    """
    #

    current_frame = 0
    bag = []

    while current_frame + frames_per_sample < number_frames: # we make sure we have enough data for slicing
        #simple array slicing on the first dimension (time/freq axis)
        sample_1 = speech_1_features[current_frame:current_frame + frames_per_sample, :]
        sample_2 = speech_2_features[current_frame:current_frame + frames_per_sample, :]
        sample_mix = speech_mix_features[current_frame:current_frame + frames_per_sample, :]
        voice_activity_detection = speech_voice_activity_detection[current_frame:current_frame + frames_per_sample, :]
        # phase = speech_phase[k: k + frames_per_sample, :]

        # Y: indicator of the belongings of the TF bin
        # 1st speaker or second speaker
        Y = np.array([sample_1 > sample_2, sample_1 < sample_2])
        # Y now will have 2-axis one for each signal comparision with other
        # returned Y will be [2, x, y], hence the transpose [x, y, 2]
        Y = np.transpose(Y, [1, 2, 0])
        bag.append((sample_mix, voice_activity_detection, Y))
        current_frame = current_frame + frames_per_sample

        sample_1 = sample_2 = sample_mix = voice_activity_detection = Y = None
    return bag

#@profile
def _get_speech_features(wav_file_1,
                         wav_file_2,
                         sampling_rate,
                         frame_size,
                         neff,
                         amp_fac,
                         min_amp,
                         threshold,
                         global_mean,
                         global_std,
                         frames_per_sample):
    """


    :param args:
    :return:
    """
    # (wav_file_1, wav_file_2, sampling_rate, frame_size, neff, amp_fac, min_amp,
    #  threshold, global_mean, global_std, frames_per_sample) = args

    try:
        # TODO: experimental though - is multi-processing required here? to reduce IO
        # print_info("{} {}".format(wav_file_1, wav_file_2))
        def _get_data():
            speech_1 = _get_speech_data(wav_file_1, sampling_rate)
            speech_2 = _get_speech_data(wav_file_2, sampling_rate)

            # print(wav_file_1, wav_file_2)

            # find the minimum length of two speeches
            length = min(len(speech_1), len(speech_2))
            #trim both the speeches to the minimum length
            speech_1 = speech_1[:length]
            speech_2 = speech_2[:length]
            # mix the signals
            speech_mix = speech_1 + speech_2

            #get the spectral features in dB
            speech_1_features = _get_log_spectrum_features(speech_1, frame_size, neff, amp_fac, min_amp)
            speech_2_features = _get_log_spectrum_features(speech_2, frame_size, neff, amp_fac, min_amp)
            speech_mix_features = _get_log_spectrum_features(speech_mix, frame_size, neff, amp_fac, min_amp)

            max_mag = np.max(speech_mix_features)
            # apply threshold to the feature signal, to find the silent portion of the signal and
            # construct a boolean array as a feature
            # https://en.wikipedia.org/wiki/Voice_activity_detection
            speech_voice_activity_detection = (speech_mix_features > (max_mag - threshold))
            # normalize the signal values with given global mean and std
            speech_mix_features_final = (speech_mix_features - global_mean) / global_std

            number_frames = speech_1_features.shape[0]

            new_data = _get_speech_samples(speech_1_features,
                                           speech_2_features,
                                           frames_per_sample,
                                           number_frames,
                                           speech_mix_features_final,
                                           speech_voice_activity_detection)
            # print_error("deleting speech_1_features, speech_2_features, speech_voice_activity_detection, speech_mix_features")
            speech_1_features = speech_2_features = speech_voice_activity_detection = speech_mix_features = speech_mix_features_final = None
            return new_data

        new_data = _get_data()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as err:
        print_error(traceback.print_exc())
        new_data = []

    return new_data

def get_features(speech_mix, VAD, Y):
    assert speech_mix.shape[0] == VAD.shape[0] == Y.shape[0]
    assert speech_mix.shape[1] == VAD.shape[1] == Y.shape[1]

    return {"speech_mix" : tf.train.Feature(float_list=tf.train.FloatList(value=speech_mix.flatten())),
            "VAD" : tf.train.Feature(int64_list=tf.train.Int64List(value=VAD.flatten())),
            "Y" : tf.train.Feature(int64_list=tf.train.Int64List(value=Y.flatten())),
            "FPS" : tf.train.Feature(int64_list=tf.train.Int64List(value=[speech_mix.shape[0]])),
            "NEFF" : tf.train.Feature(int64_list=tf.train.Int64List(value=[speech_mix.shape[1]]))}

def _generate_tf_Records(index,
                         wav_file_1,
                         wav_file_2,
                         sampling_rate,
                         frame_size,
                         amp_fac,
                         neff,
                         min_amp,
                         threshold,
                         global_mean,
                         global_std,
                         frames_per_sample,
                         out_dir):
    data = _get_speech_features(wav_file_1,
                                wav_file_2,
                                sampling_rate,
                                frame_size,
                                neff,
                                amp_fac,
                                min_amp,
                                threshold,
                                global_mean,
                                global_std,
                                frames_per_sample)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file = os.path.join(out_dir, str(index) + ".tfrecord")
    with tf.python_io.TFRecordWriter(file) as writer:
        for record in data:
            features = tf.train.Features(feature=get_features(record[0], record[1], record[2]))

            example = tf.train.Example(features=features)


            writer.write(example.SerializeToString())

    return file
