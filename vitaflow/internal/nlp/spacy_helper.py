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

import spacy
import os
from tqdm import tqdm
from tensorflow.python.platform import gfile

from vitaflow.utils.print_helper import print_warn
# from vitaflow.helpers.print_helper import print_warn
from vitaflow.deprecated.iterators import SpecialTokens


def naive_vocab_creater(out_file_name, lines, use_nlp):
    '''
    Given list of lines it extracts the vocab from each line and dumps it to the file.
    If the file already present it reads it back
    :param out_file_name: Output file name
    :param lines: List of lines
    :param use_nlp: Boolean flag to use spaCy for tokenizing
    :return: Length of the vocab, List of vocab
    '''
    if not os.path.exists(out_file_name):
        nlp = spacy.load('en_core_web_md')
        final_vocab = [SpecialTokens.PAD_WORD, SpecialTokens.UNK_WORD]
        if use_nlp:
            vocab = [word.text for line in tqdm(lines, desc="vocab_filter")
                     for word in nlp(str(line)) if word.text in nlp.vocab]
        else:
            print(lines)
            vocab = [word for line in tqdm(lines) for word in line.split(" ")]

        vocab = set(vocab)
        vocab = sorted(vocab)

        try:
            vocab.remove(SpecialTokens.UNK_WORD)
        except:
            print("No {} token found".format(SpecialTokens.UNK_WORD))

        vocab = list(vocab)
        final_vocab.extend(vocab)

        print_warn(out_file_name)

        # Create a file and store the words
        with gfile.Open(out_file_name, 'w') as f:
            for word in final_vocab:
                f.write("{}\n".format(word))
    else:
        with open(out_file_name) as file:
            lines = []
            for line in file:
                line = line.strip("\n")
                lines.append(line)
        final_vocab = list(lines)

    return len(final_vocab), final_vocab


def vocab_to_tsv(out_file_name, vocab_list):
    '''
    Stores the vocab list to a file & returns a vocab mapper(vocab -> id).
    If the file is already available, retrives it from stored file.
    :param vocab_list: List of words/characters
    :return: Character to ID mapper
    '''
    if not os.path.exists(out_file_name):
        with gfile.Open(out_file_name, 'w') as file:
            mapper = dict()
            i = 0
            for word in tqdm(vocab_list):
                if len(word) > 0:
                    file.write("{}\n".format(word))
                    mapper[word] = i
                    i += 1
        print('{} words into {}'.format(len(vocab_list), out_file_name))
        # mapper = {c: i for i, c in enumerate(vocab_list)}
    else:
        # TODO: Reading a file is costlier than re-creating if from list.
        with open(out_file_name) as file:
            # lines = []
            # for line in file:
            #     line = line.strip("\n")
            #     lines.append(line)
            mapper = {c.strip("\n"): i for i, c in enumerate(file)}
    return mapper


def get_char_vocab(words_vocab):
    '''
    Extracts characters from the list of words
    :param words_vocab: List of words
    :return: Sorted list of characters
    '''
    chars = set()
    for word in words_vocab:
        for char in word:
            chars.add(str(char))
    return sorted(chars)
