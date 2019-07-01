import gin
from vitaflow.datasets.text.conll_2003_dataset import ConllDataset2003
from vitaflow.iterators.text.csv_seq_to_seq_iterator import CSVSeqToSeqIterator
from vitaflow.models.text.seq2seq.bilstm_crf import BiLSTMCrf

@gin.configurable
def get_experiment_root_directory(value):
    return value
