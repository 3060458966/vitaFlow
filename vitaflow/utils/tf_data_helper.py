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

"""
import tensorflow as tf

import tensorflow as tf
from tensorflow.python.platform import gfile


def tf_vocab_processor(lines, out_file_name, max_doc_length=1000, min_frequency=0):
    # Create vocabulary
    # min_frequency -> consider a word if and only it repeats for given count
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length,
                                                                         min_frequency=min_frequency)
    vocab_processor.fit(lines)

    vocab = vocab_processor.vocabulary_._mapping.keys()

    # Create a file and store the words
    with gfile.Open(out_file_name, 'wb') as f:
        for word in vocab:
            f.write("{}\n".format(word))

    nwords = len(vocab_processor.vocabulary_) + 1  # <UNK>

    print('{} words into {}'.format(nwords, out_file_name))
    return (nwords, vocab)


def get_sequence_length_old(sequence):
    '''
    Returns the sequence length, droping out all the zeros if the sequence is padded
    :param sequence: Tensor(shape=[batch_size, doc_length, feature_dim])
    :return: Array of Document lengths of size batch_size
    '''
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def get_sequence_length(sequence_ids, pad_word_id=0):
    '''
    Returns the sequence length, droping out all the padded tokens if the sequence is padded

    :param sequence_ids: Tensor(shape=[batch_size, doc_length])
    :param pad_word_id: 0 is default
    :return: Array of Document lengths of size batch_size
    '''
    flag = tf.greater_equal(sequence_ids, 1)  # TODO 1 -> start of <UNK> vocab id
    used = tf.cast(flag, tf.int32)
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length
