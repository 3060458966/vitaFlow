# import os
# from multiprocessing import Pool
#
# import librosa
# import numpy as np
# from numpy.lib import stride_tricks
# from tqdm import tqdm
#
# from vitaflow.helpers.print_helper import print_error
#
# enable_multiprocessing = True
# enable_file_pickling = False
#
#
# def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
#     """
#     short time fourier transform of audio signal
#     """
#     win = window(frameSize)
#     hopSize = int(frameSize - np.floor(overlapFac * frameSize))
#     samples = np.array(sig, dtype='float64')
#     # cols for windowing
#     cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
#     # zeros at end (thus samples can be fully covered by frames)
#     samples = np.append(samples, np.zeros(frameSize))
#
#     frames = stride_tricks.as_strided(
#         samples,
#         shape=(cols, frameSize),
#         strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
#     frames *= win
#     return np.fft.rfft(frames)
#
#
# def _get_speech_samples(speech_1_features,
#                         speech_2_features,
#                         frames_per_sample,
#                         len_spec,
#                         speech_mix_features,
#                         speech_VAD):
#     """
#
#     :param speech_1_features:
#     :param speech_2_features:
#     :param frames_per_sample:
#     :param len_spec:
#     :param speech_mix_features:
#     :param speech_VAD:
#     :return:
#     """
#     # Processes one audio pair
#     k = 0
#     bag = []
#
#     while k + frames_per_sample < len_spec:
#         sample_mix = speech_mix_features[k:k + frames_per_sample, :].astype('float32')
#         VAD = speech_VAD[k:k + frames_per_sample, :].astype('bool')
#
#         sample_1 = speech_1_features[k:k + frames_per_sample, :]
#         sample_2 = speech_2_features[k:k + frames_per_sample, :]
#         # phase = speech_phase[k: k + frames_per_sample, :]
#         # Y: indicator of the belongings of the TF bin
#         # 1st speaker or second speaker
#         Y = np.array([sample_1 > sample_2,
#                       sample_1 < sample_2]
#                      ).astype('bool')
#         Y = np.transpose(Y, [1, 2, 0]).astype('bool')
#         bag.append((sample_mix, VAD, Y))
#         k = k + frames_per_sample
#     return bag
#
#
# def _get_log_spectrum_features(speech, frame_size, neff, amp_fac, min_amp):
#     """
#
#     :param speech:
#     :param frame_size:
#     :param neff:
#     :param amp_fac:
#     :param min_amp:
#     :return:
#     """
#     speech_features = np.abs(stft(speech, frame_size)[:, :neff])
#     speech_features = np.maximum(speech_features, np.max(speech_features) / min_amp)
#     speech_features = 20. * np.log10(speech_features * amp_fac)
#     return speech_features
#
#
# def _get_speech_data(wav_file, sampling_rate):
#     """
#
#     :param wav_file:
#     :param sampling_rate:
#     :return:
#     """
#     speech, _ = librosa.core.load(wav_file, sr=sampling_rate)
#     # amp factor between -3 dB - 3 dB
#     fac = np.random.rand(1)[0] * 6 - 3
#     speech = 10. ** (fac / 20) * speech
#     return speech
#
#
# def _get_speech_features(args):
#     (wav_file_1, wav_file_2, sampling_rate, frame_size, neff, amp_fac, min_amp,
#      threshold, global_mean, global_std, frames_per_sample) = args
#
#     try:
#         # TODO: experimental though - is multi-processing required here? to reduce IO
#         speech_1 = _get_speech_data(wav_file_1, sampling_rate)
#         speech_2 = _get_speech_data(wav_file_2, sampling_rate)
#
#         # print(wav_file_1, wav_file_2)
#
#         # mix
#         length = min(len(speech_1), len(speech_2))
#         speech_1 = speech_1[:length]
#         speech_2 = speech_2[:length]
#         speech_mix = speech_1 + speech_2
#
#         speech_1_features = _get_log_spectrum_features(speech_1, frame_size, neff, amp_fac, min_amp)
#         speech_2_features = _get_log_spectrum_features(speech_2, frame_size, neff, amp_fac, min_amp)
#
#         # same for the mixture
#         speech_mix_spec0 = stft(speech_mix, frame_size)[:, :neff]
#         speech_mix_features = np.abs(speech_mix_spec0)
#
#         # speech_phase = speech_mix_spec0 / speech_mix_spec
#         speech_mix_features = np.maximum(speech_mix_features, np.max(speech_mix_features) / min_amp)
#         speech_mix_features = 20. * np.log10(speech_mix_features * amp_fac)
#         max_mag = np.max(speech_mix_features)
#
#         speech_VAD = (speech_mix_features > (max_mag - threshold)).astype(int)
#         speech_mix_features = (speech_mix_features - global_mean) / global_std
#
#         len_spec = speech_1_features.shape[0]
#
#         new_data = _get_speech_samples(speech_1_features, speech_2_features, frames_per_sample, len_spec,
#                                        speech_mix_features, speech_VAD)
#     except Exception as err:
#         t = '-------------------------\n'
#         print_error(err)
#         print('{} Failed to Run: _get_speech_features {} {} \n {}'.format(t, wav_file_1, wav_file_2, t))
#         print('---' * 5)
#         new_data = []
#
#     return new_data
#
#
# def yield_samples_multicore(speaker_file_match,
#                             sampling_rate,
#                             frame_size,
#                             amp_fac,
#                             neff,
#                             min_amp,
#                             threshold,
#                             global_mean,
#                             global_std,
#                             frames_per_sample,
#                             num_threads,
#                             tqdm_desc):
#
#     # for each file pair, generate their mixture and reference samples
#
#     p = Pool(num_threads)
#     input_params = [(wav_file_1, wav_file_2, sampling_rate, frame_size, neff, amp_fac, min_amp,
#                      threshold, global_mean, global_std, frames_per_sample) for (wav_file_1, wav_file_2) in
#                     speaker_file_match.items()]
#     with tqdm(total=len(speaker_file_match.items())) as pbar:
#         for i, res in tqdm(enumerate(p.imap_unordered(_get_speech_features, input_params)), desc=tqdm_desc):
#             pbar.update()
#             for data in res:
#                 yield data
#
