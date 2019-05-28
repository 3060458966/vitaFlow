# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
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
A model class that uses BiLSTM word and char level embeddings
"""

import os

import tensorflow as tf
from overrides import overrides
from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys

import gin
from vitaflow.internal.features.feature_types import ITextFeature
from vitaflow.internal.models.model_base import ModelBase
from vitaflow.iterators.text.csv_seq_to_seq_iterator import CSVSeqToSeqIterator
from vitaflow.iterators.text.vocabulary import SpecialTokens
from vitaflow.utils.print_helper import *
from vitaflow.utils.tf_data_helper import get_sequence_length


@gin.configurable
class BiLSTMCrf(ModelBase, ITextFeature):
    """

    .. code-block:: text

        Features : {Sentence, [character ids]}, label
        Word Feature Tensor
        Character Feature Tensor
        Word Embeddings Layer
        Character Embeddings Layer
        Word Level BiLSTM
        Character Level BiLSTM
        Word Level BiLSTM Output + Character Level BiLSTM
        Fully Connected Layer
        CRF Loss
        Classes and Probability Predictions


    References:
        - https://github.com/guillaumegenthial/sequence_tagging
        - https://github.com/jiaqianghuai/tf-lstm-crf-batch
        - https://www.tensorflow.org/api_docs/python/tf/contrib/crf
        - https://github.com/Franck-Dernoncourt/NeuroNER
        - https://www.clips.uantwerpen.be/conll2003/ner/
        - https://stackoverflow.com/questions/3330227/free-tagged-corpus-for-named-entity-recognition

        - https://sites.google.com/site/ermasoftware/getting-started/ne-tagging-conll2003-data
        - Dataset: https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
        - Reference: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/estimators/abalone.py
        - https://github.com/tensorflow/tensorflow/issues/14018

    """

    def __init__(self,
                 model_root_directory,
                 experiment_name,
                 seperator,
                 use_char_embd=False,
                 learning_rate=0.001,
                 word_level_lstm_hidden_size=24,
                 char_level_lstm_hidden_size=24,
                 word_emd_size=24,
                 char_emd_size=24,
                 num_lstm_layers=1,
                 keep_probability=0.5,
                 data_iterator: CSVSeqToSeqIterator = None):
        ITextFeature.__init__(self)
        ModelBase.__init__(self,
                           experiment_name=experiment_name,
                           model_root_directory=model_root_directory)

        # if not isinstance(data_iterator, CoNLLCsvDataIterator):
        #     raise RuntimeError

        # Constant params
        # self.UNKNOWN_WORD = SpecialTokens.UNK_WORD
        # self.PAD_WORD = SpecialTokens.PAD_WORD
        self.SEPERATOR = seperator

        # Preprocessing Paramaters
        self.TAGS_VOCAB_FILE = data_iterator.ENTITY_VOCAB_FILE
        self.WORDS_VOCAB_FILE = data_iterator.WORDS_VOCAB_FILE
        self.CHARS_VOCAB_FILE = data_iterator.CHARS_VOCAB_FILE

        self.VOCAB_SIZE = data_iterator.word_vocab_size
        self.CHAR_VOCAB_SIZE = data_iterator.char_vocab_size
        self.NUM_TAGS = data_iterator.num_lables

        # Model hyper parameters
        self.USE_CHAR_EMBEDDING = use_char_embd
        self.LEARNING_RATE = learning_rate
        self.KEEP_PROP = keep_probability
        self.WORD_EMBEDDING_SIZE = word_emd_size
        self.CHAR_EMBEDDING_SIZE = char_emd_size
        self.WORD_LEVEL_LSTM_HIDDEN_SIZE = word_level_lstm_hidden_size
        self.CHAR_LEVEL_LSTM_HIDDEN_SIZE = char_level_lstm_hidden_size
        self.NUM_LSTM_LAYERS = num_lstm_layers

    def _build_layers(self, features, mode):

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        text_features = features[self.FEATURE_1_NAME]

        if self.USE_CHAR_EMBEDDING:
            # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
            char_ids = features[self.FEATURE_2_NAME]

            tf.logging.info('char_ids: =======> {}'.format(char_ids))

            s = tf.shape(char_ids)

            # remove pad words
            char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2]))  # 20 -> char dim

        with tf.variable_scope("sentence-words-2-ids"):
            word_table = lookup.index_table_from_file(vocabulary_file=self.WORDS_VOCAB_FILE,
                                                      num_oov_buckets=0,  # TODO use this for Out of Vocab
                                                      default_value=SpecialTokens.UNK_WORD_ID,
                                                      # id of <UNK>  w.r.t WORD VOCAB
                                                      name="table")
            tf.logging.info('word_table info: {}'.format(word_table))

            # [BATCH_SIZE, 1]
            words = tf.string_split(text_features, delimiter=self.SEPERATOR)

            # [BATCH_SIZE, ?] i.e [BATCH_SIZE, VARIABLE_SEQ_LENGTH]
            densewords = tf.sparse_tensor_to_dense(words,
                                                   default_value=SpecialTokens.PAD_WORD)  # TODO add test case

            # [BATCH_SIZE, ?] i.e [BATCH_SIZE, MAX_SEQ_LENGTH]
            token_ids = word_table.lookup(densewords)  # TODO check is it variable length or not?

        with tf.variable_scope("word-embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=self.VOCAB_SIZE,
                                                               embed_dim=self.WORD_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=self.KEEP_PROP,
                                                seed=42,
                                                training=mode == tf.estimator.ModeKeys.TRAIN)

            # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
            tf.logging.info('word_embeddings =====> {}'.format(word_embeddings))

            # seq_length = get_sequence_length_old(word_embeddings) TODO working
            # [BATCH_SIZE, ]
            seq_length = get_sequence_length(token_ids)
            self.seq_length = seq_length
            tf.logging.info('seq_length =====> {}'.format(seq_length))

        with tf.variable_scope("char_embed_layer"):
            if self.USE_CHAR_EMBEDDING:
                print_error((self.CHAR_VOCAB_SIZE, self.CHAR_EMBEDDING_SIZE))
                char_embeddings = tf.contrib.layers.embed_sequence(char_ids,
                                                                   vocab_size=self.CHAR_VOCAB_SIZE,
                                                                   embed_dim=self.CHAR_EMBEDDING_SIZE,
                                                                   initializer=tf.contrib.layers.xavier_initializer(
                                                                       seed=42))

                # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
                char_embeddings = tf.layers.dropout(char_embeddings,
                                                    rate=self.KEEP_PROP,
                                                    seed=42,
                                                    training=mode == tf.estimator.ModeKeys.TRAIN)  # TODO add test case

                tf.logging.info('char_embeddings =====> {}'.format(char_embeddings))

        with tf.variable_scope("chars_level_bilstm_layer"):
            if self.USE_CHAR_EMBEDDING:
                # put the time dimension on axis=1
                shape = tf.shape(char_embeddings)

                BATCH_SIZE = shape[0]
                MAX_DOC_LENGTH = shape[1]
                CHAR_MAX_LENGTH = shape[2]

                TOTAL_DOCS_LENGTH = tf.reduce_sum(seq_length)

                # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]  ===>
                #      [BATCH_SIZE * MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[BATCH_SIZE * MAX_DOC_LENGTH, CHAR_MAX_LENGTH,
                                                    self.CHAR_EMBEDDING_SIZE],
                                             name="reduce_dimension_1")

                tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

                # word_lengths = get_sequence_length_old(char_embeddings) TODO working
                word_lengths = get_sequence_length(char_ids_reshaped)

                tf.logging.info('word_lengths =====> {}'.format(word_lengths))

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                                  state_is_tuple=True)

                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    dtype=tf.float32,
                    sequence_length=word_lengths,
                    inputs=char_embeddings,
                    scope="encode_words")

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                encoded_words = tf.concat([output_fw, output_bw], axis=-1)

                # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
                encoded_words = tf.reshape(encoded_words,
                                           shape=[BATCH_SIZE, MAX_DOC_LENGTH, 2 *
                                                  self.CHAR_LEVEL_LSTM_HIDDEN_SIZE])

                tf.logging.info('encoded_words =====> {}'.format(encoded_words))

        with  tf.variable_scope("word_level_lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(self.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(self.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=self.KEEP_PROP)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=self.KEEP_PROP)
            else:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  self.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  self.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)

            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype=tf.float32,
                sequence_length=seq_length,
                inputs=word_embeddings,
                scope="encod_sentence")

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE) TODO check MAX_SEQ_LENGTH?
            encoded_sentence = tf.concat([fw_output_one,
                                          bw_output_one], axis=-1)

            tf.logging.info('encoded_sentence =====> {}'.format(encoded_sentence))

        with tf.variable_scope("char_word_embeddings-mergeing_layer"):
            if self.USE_CHAR_EMBEDDING:
                encoded_doc = tf.concat([encoded_words, encoded_sentence], axis=-1, name="sentence_words_concat")
            else:
                encoded_doc = encoded_sentence

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE + 2*CHAR_LEVEL_LSTM_HIDDEN_SIZE]
            encoded_doc = tf.layers.dropout(encoded_doc,
                                            rate=self.KEEP_PROP,
                                            seed=42,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

        with tf.variable_scope("projection"):

            NUM_WORD_LSTM_NETWORKS = 1 + 1  # word_level_lstm_layer BiDirectional
            NUM_CHAR_LSTM_NETWORKS = 1 + 1  # char_level_lstm_layer BiDirectional

            # Example: If WORD_LEVEL_LSTM_HIDDEN_SIZE = 300, CHAR_LEVEL_LSTM_HIDDEN_SIZE = 300,
            # NEW_SHAPE = 2 * 300 + 2 * 300 = 1200
            NEW_SHAPE = NUM_WORD_LSTM_NETWORKS * self.WORD_LEVEL_LSTM_HIDDEN_SIZE + \
                        NUM_CHAR_LSTM_NETWORKS * self.CHAR_LEVEL_LSTM_HIDDEN_SIZE

            if self.USE_CHAR_EMBEDDING:
                # [NEW_SHAPE, NUM_TAGS]
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[NEW_SHAPE, self.NUM_TAGS])
                # [NUM_TAGS]
                b = tf.get_variable("b", shape=[self.NUM_TAGS],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                # [NUM_WORD_LSTM_NETWORKS * WORD_LEVEL_LSTM_HIDDEN_SIZE, NUM_TAGS]
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[NUM_WORD_LSTM_NETWORKS * self.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                           self.NUM_TAGS])
                # [NUM_TAGS]
                b = tf.get_variable("b", shape=[self.NUM_TAGS],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
            # [MAX_SEQ_LENGTH]
            nsteps = tf.shape(encoded_doc)[1]

            tf.logging.info("nsteps: =====> {} ".format(nsteps))

            if self.USE_CHAR_EMBEDDING:
                encoded_doc = tf.reshape(encoded_doc, [-1, NEW_SHAPE],
                                         name="reshape_encoded_doc")
            else:
                encoded_doc = tf.reshape(encoded_doc,
                                         [-1, NUM_WORD_LSTM_NETWORKS * self.WORD_LEVEL_LSTM_HIDDEN_SIZE],
                                         name="reshape_encoded_doc")

            tf.logging.info("encoded_doc: {}".format(encoded_doc))
            encoded_doc = tf.matmul(encoded_doc, W) + b

            tf.logging.info("encoded_doc: {}".format(encoded_doc))
            # [BATCH_SIZE, MAX_SEQ_LENGTH, NUM_TAGS]
            logits = tf.reshape(encoded_doc, [-1, nsteps, self.NUM_TAGS], name="reshape_predictions")
            tf.logging.info("logits: {}".format(logits))
            return logits

    def _get_loss(self, labels, logits):
        with  tf.variable_scope("loss-layer"):
            """Defines the loss"""

            # if mode == ModeKeys.INFER:
            #     ner_ids = tf.placeholder(tf.int32, shape=[None, None],
            #                              name="labels")  # no labels during prediction
            # else:
            ner_ids = labels

            print_error(self.seq_length)
            print_error(logits)
            print_error(ner_ids)
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                logits, ner_ids, self.seq_length)

            tf.logging.info("log_likelihood:  =====> {}".format(log_likelihood))

            # [NUM_TAGS, NUM_TAGS]
            trans_params = trans_params  # need to evaluate it for decoding
            tf.logging.info("trans_params: =====> {}".format(trans_params))
            ner_crf_loss = tf.reduce_mean(-log_likelihood)

            tf.summary.scalar("loss", ner_crf_loss)

            return ner_crf_loss, trans_params

    def _get_predictions(self, logits, trans_params):
        viterbi_seq, best_score = tf.contrib.crf.crf_decode(logits, trans_params, self.seq_length)

        tf.logging.info("viterbi_seq: {}".format(viterbi_seq))

        predictions = {  # TODO features class
            "classes": tf.cast(tf.argmax(logits, axis=-1),
                               tf.int32),
            # [BATCH_SIZE, SEQ_LEN]
            "viterbi_seq": viterbi_seq,
            # [BATCH_SIZE]
            "confidence": tf.reduce_max(tf.nn.softmax(logits, dim=-1), axis=-1),

            "top_3_indices": tf.nn.top_k(tf.nn.softmax(logits, dim=-1), k=3).indices,

            "top_3_confidence": tf.nn.top_k(tf.nn.softmax(logits, dim=-1), k=3).values
        }

        return predictions

    def _get_optimizer(self, loss):
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=self.LEARNING_RATE)
        return train_op

    def _get_eval_metrics(self, predictions, labels):
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["viterbi_seq"],
                name='accuracy'),
            'Precision': tf.metrics.precision(
                labels=labels,
                predictions=predictions["viterbi_seq"],
                name='Precision'),
            'Recall': tf.metrics.recall(
                labels=labels,
                predictions=predictions["viterbi_seq"],
                name='Recall')
        }

    @overrides
    def _build(self, features, labels, params, mode, config=None):

        # Loss, training and eval operations are not needed during inference.
        loss = None
        optimizer = None
        eval_metric_ops = {}

        with tf.variable_scope("ner-tags-2-ids"):
            if mode != ModeKeys.INFER:
                ner_table = lookup.index_table_from_file(vocabulary_file=self.TAGS_VOCAB_FILE,
                                                         num_oov_buckets=0,
                                                         default_value=SpecialTokens.UNK_TAG_ID,  # id of <UNK> w.r.t ENTITY VOCAB
                                                         name="table")

                tf.logging.info('ner_table info: {}'.format(ner_table))

                # [BATCH_SIZE, 1]
                labels_splitted = tf.string_split(labels, delimiter=self.SEPERATOR)
                # [BATCH_SIZE, ?] i.e [BATCH_SIZE, VARIABLE_SEQ_LENGTH]
                labels_splitted_dense = tf.sparse_tensor_to_dense(labels_splitted,
                                                                  default_value="O")
                # [BATCH_SIZE, ?] i.e [BATCH_SIZE, MAX_SEQ_LENGTH]
                ner_ids = ner_table.lookup(labels_splitted_dense)
                ner_ids = tf.cast(ner_ids, tf.int32)

                tf.logging.info("ner_ids: {}".format(ner_ids))

        logits = self._build_layers(features=features, mode=mode)

        if mode == ModeKeys.INFER:
            ner_ids = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")  # no labels during prediction
        else:
            ner_ids = ner_ids

        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logits, ner_ids, self.seq_length)

        tf.logging.info("log_likelihood:  =====> {}".format(log_likelihood))

        # [NUM_TAGS, NUM_TAGS]
        trans_params = trans_params  # need to evaluate it for decoding
        tf.logging.info("trans_params: =====> {}".format(trans_params))

        predictions = self._get_predictions(logits=logits, trans_params=trans_params)

        eval_metric_ops = {}

        if mode != tf.estimator.ModeKeys.PREDICT:
            # labels = tf.reshape(labels, shape=(-1, self._out_dim), name="labels")
            tf.logging.info('labels: -----> {}'.format(ner_ids))

            loss = tf.reduce_mean(-log_likelihood, name="crf_loss")
            optimizer = self._get_optimizer(loss)
            eval_metric_ops = self._get_eval_metrics(predictions=predictions, labels=ner_ids)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=eval_metric_ops)
