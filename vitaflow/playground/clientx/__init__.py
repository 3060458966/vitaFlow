import gin
import os
from vitaflow.playground.clientx.clientx_dataset import CLIENTXDataset
from vitaflow.iterators.text.csv_seq_to_seq_iterator import CSVSeqToSeqIterator
from vitaflow.models.text.seq2seq import BiLSTMCrf

BiLSTMCrf
CSVSeqToSeqIterator

@gin.configurable
def get_experiment_root_directory(value):
    # Check if demo is running- Set only in demo mode
    if "DEMO_DATA_PATH" in os.environ:
        return os.environ["DEMO_DATA_PATH"]
    else:
        return  value