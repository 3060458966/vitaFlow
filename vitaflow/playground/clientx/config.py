import os

experiment_root_directory = os.environ['DEMO_DATA_PATH']
experiment_name = "clientx_dataset"
use_char_embd = True

experiments = {
    "num_epochs": 150,
    "dataset_class_with_path": "vitaflow.playground.clientx.clientx_dataset.CLIENTXDataset",
    "iterator_class_with_path": "vitaflow.iterators.text.csv_seq_to_seq_iterator.CSVSeqToSeqIterator",
    "model_class_with_path": "vitaflow.models.text.seq2seq.BiLSTMCrf",
    "save_checkpoints_steps" : 50,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : 25,
    "log_step_count_steps" : 10,

    # dataset - details
    "vitaflow.playground.clientx.clientx_dataset.CLIENTXDataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "minimum_num_words": 5,
        "over_write": False,
    },

    # data iterator
    "vitaflow.iterators.text.csv_seq_to_seq_iterator.CSVSeqToSeqIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "clientx_data_iterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "text_col": "0",
        "entity_col": "1",
        "batch_size": 16,
        "seperator": "~",  # potential error point depending on the dataset
        "quotechar": "^",
        "max_word_length": 20,
        "use_char_embd": use_char_embd
    },

    # data model
    "vitaflow.models.text.seq2seq.BiLSTMCrf": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "use_char_embd": use_char_embd,
        "learning_rate": 0.001,
        "word_level_lstm_hidden_size": 8,
        "char_level_lstm_hidden_size": 4,
        "word_emd_size": 8,
        "char_emd_size": 4,
        "num_lstm_layers": 1,
        "keep_probability": 0.4
    }
}
