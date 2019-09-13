import csv
import os

import pandas as pd
import sklearn_crfsuite


# experiment_root_directory = os.environ['DEMO_DATA_PATH']  # os.path.join("/opt/data", "vitaFlow-clientx/")
# experiment_name = "clientx_dataset"
# use_char_embd = True
#
# experiments = {
#     "num_epochs": 150,
#     "dataset_class_with_path": "examples.clientx.clientx_dataset.CLIENTXDataset",
#     "iterator_class_with_path": "vitaflow.data.text.iterators.CSVSeqToSeqIterator",
#     "model_class_with_path": "vitaflow.models.text.seq2seq.BiLSTMCrf",
#     "save_checkpoints_steps": 50,
#     "keep_checkpoint_max": 5,
#     "save_summary_steps": 25,
#     "log_step_count_steps": 10,
#
#     # dataset - details
#     "examples.clientx.clientx_dataset.CLIENTXDataset": {
#         "experiment_root_directory": experiment_root_directory,
#         "experiment_name": experiment_name,
#         "preprocessed_data_path": "preprocessed_data",
#         "train_data_path": "train",
#         "validation_data_path": "val",
#         "test_data_path": "test",
#         "minimum_num_words": 5,
#         "over_write": False,
#     },
#
#     # data iterator
#     "vitaflow.data.text.iterators.CSVSeqToSeqIterator": {
#         "experiment_root_directory": experiment_root_directory,
#         "experiment_name": experiment_name,
#         "iterator_name": "clientx_data_iterator",
#         "preprocessed_data_path": "preprocessed_data",
#         "train_data_path": "train",
#         "validation_data_path": "val",
#         "test_data_path": "test",
#         "text_col": "0",
#         "entity_col": "1",
#         "batch_size": 16,
#         "seperator": "~",  # potential error point depending on the dataset
#         "quotechar": "^",
#         "max_word_length": 20,
#         "use_char_embd": use_char_embd
#     },
#
#     # data model
#     "vitaflow.models.text.seq2seq.BiLSTMCrf": {
#         "model_root_directory": experiment_root_directory,
#         "experiment_name": experiment_name,
#         "use_char_embd": use_char_embd,
#         "learning_rate": 0.001,
#         "word_level_lstm_hidden_size": 8,
#         "char_level_lstm_hidden_size": 4,
#         "word_emd_size": 8,
#         "char_emd_size": 4,
#         "num_lstm_layers": 1
#     }
# }


def strip_iob(iob_tag):
    tag = iob_tag.replace("B-", "")
    tag = tag.replace("I-", "")
    return tag


def is_new_tag(prev, current):
    if "O" in prev:
        prev_t, prev_w = "O", "O"
    else:
        prev_t, prev_w = prev.split("-")
    if "O" in current:
        current_t, current_w = "O", "O"
    else:
        current_t, current_w = current.split("-")

    if prev_w != current_w:
        return True
    else:
        if prev_t == "B" and current_t == "I":
            return False
        elif prev_t == "I" and current_t == "B":
            return True
        elif prev_t == "I" and current_t == "I":
            return False
        else:
            return False


# full_path = os.environ['DEMO_DATA_PATH']
full_path = '/Users/sampathm/devbox/vitaFlow/vitaflow/standoff2conll/receipt_mock_data'

train_sent = []
for i in os.listdir(full_path + "/train"):
    df = pd.read_csv(os.path.join(full_path + "/train", i), sep="\t", quoting=csv.QUOTE_NONE)
    data = list(zip(*[df[c].values.tolist() for c in ['0', '1']]))
    train_sent.append(data)

test_sent = []
for i in os.listdir(full_path + "/val"):
    df = pd.read_csv(os.path.join(full_path + "/val", i), sep="\t", quoting=csv.QUOTE_NONE)
    data = list(zip(*[df[c].values.tolist() for c in ['0', '1']]))
    test_sent.append(data)

predict_sent = []
for i in os.listdir(full_path + "/test/"):
    df = pd.read_csv(full_path + "/test/" + i, sep="\t", quoting=csv.QUOTE_NONE)
    data = list(zip(*[df[c].values.tolist() for c in ['0', '1']]))
    predict_sent.append(data)

train_sents = train_sent
test_sents = test_sent


def word2features(sent, i):
    word = sent[i][0]
    #     postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower() if type(word) is str else False,
        'word[-3:]': word[-3:] if type(word) is str else word,
        'word[-2:]': word[-2:] if type(word) is str else word,
        'word.isupper()': word.isupper() if type(word) is str else False,
        'word.istitle()': word.istitle() if type(word) is str else False,
        'word.isdigit()': word.isdigit() if type(word) is str else False,
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower() if type(word1) is str else False,
            '-1:word.istitle()': word1.istitle() if type(word1) is str else False,
            '-1:word.isupper()': word1.isupper() if type(word1) is str else False,
            # '-1:postag': postag1,
            # '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower() if type(word1) is str else False,
            '+1:word.istitle()': word1.istitle() if type(word1) is str else False,
            '+1:word.isupper()': word1.isupper() if type(word1) is str else False,
            # '+1:postag': postag1,
            # '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.06,
    c2=0.3,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

X_predict = [sent2features(s) for s in predict_sent]

predictions = crf.predict(X_predict)

predict_sent1 = []
predict_sent2 = []
for i in predict_sent:
    for j in i:
        predict_sent1.append(j[0])
    predict_sent2.append(predict_sent1)
    predict_sent1 = []

l = []
for i in range(len(predictions)):
    l.append(list(zip(predictions[i], predict_sent2[i])))

for i in os.listdir(full_path + "/test/"):
    df = pd.read_csv(full_path + "/test/" + i, sep="\t", quoting=csv.QUOTE_NONE)

count = 0
for i in os.listdir(full_path + "/test/"):
    with open(full_path + "/test/" + i.rsplit(".", 1)[0] + ".csv", mode='wt',
              encoding='utf-8') as myfile:
        csv_out = csv.writer(myfile)
        csv_out.writerow(["word", "pred"])
        for row in l[count]:
            csv_out.writerow([str(row[1]), str(row[0])])
    count = count + 1

for i in os.listdir(full_path + "/test/"):
    df = pd.read_csv(full_path + "/test/" + i.rsplit(".", 1)[0] + ".csv")

doc_text = ""
for index, row in df.iterrows():
    if row["pred"] != "O":
        if index == 0 or not enter:
            text = row["word"]
            prev_tag = row["pred"]
            enter = True

        else:
            # second index onwards
            if is_new_tag(prev_tag, row["pred"]):
                doc_text = doc_text + text + "~" + strip_iob(prev_tag) + "\n"
                text = row["word"]

            else:
                text = text + " " + row["word"]
            prev_tag = row["pred"]
doc_text = doc_text + text + "~" + strip_iob(prev_tag) + "\n"

with open("postprocessed/" + i.rsplit(".", 1)[0] + ".csv", "w") as post_file:
    post_file.write("Item~Tag\n")
    post_file.write(doc_text)

if __name__ == "__main__":
    print('Hahaha,.. lakakalakaklaka')
