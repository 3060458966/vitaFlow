# import sys
# import os
#
# from overrides import overrides
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import ntpath
# import traceback
# import tensorflow as tf
#
# from vitaflow.config.hyperparams import HParams
# from vitaflow.data.text.iterators.internal.dataset_types import ICSVSeq2SeqType1
# from vitaflow.data.text.iterators.internal.iconll_iterator import ICoNLLIterator
# from vitaflow.data.text.iterators.internal.feature_types import ITextFeature
#
# from vitaflow.utils.tf_hooks.data_initializers import DataIteratorInitializerHook
# from vitaflow.data.text.vocabulary import SpecialTokens
# from vitaflow.helpers.os_helper import check_n_makedirs
#
# from vitaflow.helpers.print_helper import *
# from vitaflow.data.text.nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv
#
# __all__ = ["CoNLLCsvDataIterator"]
# __version__ = '0.0.1'
# __author__ = 'Mageswaran Dhandapani <mageswaran1989@gmail.com>'
#
# class CoNLLCsvDataIterator(ICoNLLIterator, ITextFeature):
#     def __init__(self, hparams=None, dataset=None):
#         ICoNLLIterator.__init__(self, hparams=hparams, dataset=dataset)
#         ITextFeature.__init__(self)
#         self._hparams = HParams(hparams, self.default_hparams())
#
#         self._extract_vocab()
#         self.num_training_samples = -1
#
#     @staticmethod
#     def default_hparams():
#         """
#         .. role:: python(code)
#            :language: python
#
#         .. code-block:: python
#
#             {
#
#             }
#
#         Here:
#
#
#         :return: A dictionary of hyperparameters with default values
#         """
#
#         hparams = ICoNLLIterator.default_hparams()
#
#         hparams.update({
#             "seperator" : "~", # potential error point depending on the dataset
#             "quotechar" : "^",
#             "empty_line_filler" : "<LINE_END>",
#             "max_word_length" : 20,
#             "use_char_embd" : False
#         })
#
#         return hparams
#
#     def set_dataset(self, dataset):
#         if not isinstance(dataset, ICSVSeq2SeqType1):
#             raise RuntimeError("Given Dataset cant be used with this iterator")
#         self._dataset: ICSVSeq2SeqType1 = dataset
#
#     def _extract_vocab(self):
#         if not os.path.exists(self.WORDS_VOCAB_FILE) \
#                 or not os.path.exists(self.ENTITY_VOCAB_FILE) \
#                 or not os.path.exists(self.CHARS_VOCAB_FILE):
#             print_info("Preparing the vocab for the text col: {}".format(self._hparams.text_col))
#
#             lines = set()
#             entities = set()
#
#             for df_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH), desc="mergining lines"):
#                 df_file = os.path.join(self.TRAIN_FILES_IN_PATH, df_file)
#
#                 if df_file.endswith(".csv"):
#                     df = pd.read_csv(df_file).fillna(SpecialTokens.UNK_WORD)
#                 else:
#                     raise RuntimeError
#
#                 lines.update(set(df[self._hparams.text_col].values.tolist()))
#                 entities.update(set(df[self._hparams.entity_col].values.tolist()))
#
#             self.WORD_VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=lines,
#                                                                     out_file_name=self.WORDS_VOCAB_FILE,
#                                                                     use_nlp=True)
#
#             print_info("Preparing the character vocab for the text col: {}".format(self._hparams.text_col))
#
#             # Get char level vocab
#             char_vocab = [SpecialTokens.PAD_CHAR, SpecialTokens.UNK_CHAR]
#             _vocab = get_char_vocab(words_vocab)
#             char_vocab.extend(_vocab)
#
#             # Create char2id map
#             self.CHAR_2_ID_MAP = vocab_to_tsv(vocab_list=char_vocab,
#                                               out_file_name=self.CHARS_VOCAB_FILE)
#
#             self.CHAR_VOCAB_SIZE = len(self.CHAR_2_ID_MAP)
#
#             print_info("Preparing the vocab for the entity col: {}".format(self._hparams.entity_col))
#
#             # NUM_TAGS, tags_vocab = tf_vocab_processor(lines, ENTITY_VOCAB_FILE)
#             self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=entities,
#                                                             out_file_name=self.ENTITY_VOCAB_FILE,
#                                                             use_nlp=False)
#         else:
#             print_info("Reusing the vocab")
#             self.WORD_VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=None,
#                                                                     out_file_name=self.WORDS_VOCAB_FILE,
#                                                                     use_nlp=None)
#             self.CHAR_2_ID_MAP = vocab_to_tsv(out_file_name=self.CHARS_VOCAB_FILE,
#                                               vocab_list=None)
#             self.CHAR_VOCAB_SIZE = len(self.CHAR_2_ID_MAP)
#
#             self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=None,
#                                                             out_file_name=self.ENTITY_VOCAB_FILE,
#                                                             use_nlp=False)
#
#         self.TAGS_2_ID =  {id_num: tag for id_num, tag in enumerate(tags_vocab)}
#
#     @property
#     def word_vocab_size(self):
#         return self.WORD_VOCAB_SIZE
#
#     @property
#     def char_vocab_size(self):
#         return self.CHAR_VOCAB_SIZE
#
#     @property
#     def num_lables(self):
#         return self.NUM_TAGS
#
#     @property
#     def num_train_samples(self):
#         return len(self._train_files_path)
#
#     def __pad_sequences(self, sequences, pad_tok, max_length):
#         '''
#
#         :param sequences:
#         :param pad_tok:
#         :param max_length:
#         :return:
#         '''
#         sequence_padded, sequence_length = [], []
#
#         for seq in sequences:
#             seq = list(seq)
#             seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
#             sequence_padded += [seq_]
#             sequence_length += [min(len(seq), max_length)]
#
#         return sequence_padded, sequence_length
#
#     def _pad_sequences(self, sequences, pad_tok, nlevels, max_word_length=10):
#         '''
#
#         :param sequences:
#         :param pad_tok:
#         :param nlevels:
#         :param max_word_length:
#         :return:
#         '''
#         if nlevels == 1:
#             sequence_padded = []
#             sequence_length = []
#             max_length = max(map(lambda x: len(x.split(self._hparams.seperator)), sequences))
#             # sequence_padded, sequence_length = _pad_sequences(sequences,
#             #                                                   pad_tok, max_length)
#             # breaking the code to pad the string instead on its ids
#             for seq in sequences:
#                 current_length = len(seq.split(self._hparams.seperator))
#                 diff = max_length - current_length
#                 pad_data = pad_tok * diff
#                 sequence_padded.append(seq + pad_data)
#                 sequence_length.append(max_length)  # assumed
#
#         elif nlevels == 2:
#             # max_length_word = max([max(map(lambda x: len(x), seq))
#             #                        for seq in sequences])
#             sequence_padded, sequence_length = [], []
#             for seq in tqdm(sequences):
#                 # all words are same length now
#                 sp, sl = self.__pad_sequences(seq, pad_tok, max_word_length)
#                 sequence_padded += [sp]
#                 sequence_length += [sl]
#
#             max_length_sentence = max(map(lambda x: len(x), sequences))
#             sequence_padded, _ = self.__pad_sequences(sequence_padded,
#                                                       [pad_tok] * max_word_length,
#                                                       max_length_sentence)
#             sequence_length, _ = self.__pad_sequences(sequence_length, 0,
#                                                       max_length_sentence)
#
#         return np.array(sequence_padded), sequence_length
#
#     @property
#     def _train_files_path(self):
#         files = []
#         for csv_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH)):
#             csv_file = os.path.join(self.TRAIN_FILES_IN_PATH, csv_file)
#             files.append(csv_file)
#         return files
#
#     @property
#     def _val_files_path(self):
#         files = []
#         for csv_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH)):
#             csv_file = os.path.join(self.TRAIN_FILES_IN_PATH, csv_file)
#             files.append(csv_file)
#         return files
#
#
#
#     @overrides
#     def _user_map_func(self, file_path):
#         """
#         Function that maps the audio files into features and labels as on-hot vector
#         :param file_path:
#         :param label:
#         :return:
#         """
#         file_path = str(file_path, "utf-8")
#         df = pd.read_csv(file_path).fillna(SpecialTokens.UNK_WORD)
#
#         list_text = df[self._hparams.text_col].astype(str).values.tolist()
#         list_char_ids = [[self.CHAR_2_ID_MAP.get(c, 0) for c in str(word)] for word in list_text]
#         list_tag = df[self._hparams.entity_col].astype(str).values.tolist()
#
#         if self._hparams.use_char_embd:
#             # space is used so that it can append to the string sequence
#             sentence_feature1, seq_length = \
#                 self._pad_sequences(list_text,
#                                     nlevels=1,
#                                     pad_tok="{}{}".format(self._hparams.seperator, SpecialTokens.PAD_WORD))
#             sentence_feature1 = np.array(sentence_feature1)
#
#             char_ids_feature2, seq_length = self._pad_sequences(list_char_ids,
#                                                                 nlevels=2,
#                                                                 pad_tok=int(SpecialTokens.PAD_CHAR_ID))
#             char_ids_feature2 = np.array(char_ids_feature2)
#             seq_length = np.array(seq_length)
#             tag_label, seq_length = self._pad_sequences(list_tag,
#                                                         nlevels=1,
#                                                         pad_tok="{}{}".format(self._hparams.seperator, SpecialTokens.PAD_WORD))
#             # tag_label = np.array(tag_label)
#
#             return {self.FEATURE_1_NAME : sentence_feature1, self.FEATURE_2_NAME: char_ids_feature2}, tag_label
#
#         else:
#             sentence_feature1 = list_text
#             tag_label = list_tag
#             # return {self.FEATURE_1_NAME : sentence_feature1, self.FEATURE_2_NAME: None}, tag_label
#
#             return sentence_feature1,  [""], tag_label
#
#     @overrides
#     def _user_resize_func(self, feature1, feature2, label):
#         """
#         Function that sets up the sizes of the tensor, after execution of `tf.py_func` call
#         :param data:
#         :param label:
#         :return:
#         """
#         # data = tf.reshape(data, shape=[128,33])
#         # label = tf.reshape(label, shape=[42])
#         return feature1, feature2, label
#         # return ({self.FEATURE_1_NAME: feature1, self.FEATURE_2_NAME: feature2}, label)
