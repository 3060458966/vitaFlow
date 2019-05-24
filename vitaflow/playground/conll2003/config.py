import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "conll_2003_dataset"
use_char_embd = True

experiments = {
    "num_epochs": 5,
    "dataset_class_with_path": "examples.conll2003.conll_2003_dataset.CoNLL2003Dataset",
    "iterator_class_with_path": "vitaflow.data.text.iterators.CSVSeqToSeqIterator",
    "model_class_with_path": "vitaflow.models.text.seq2seq.BiLSTMCrf",
    "save_checkpoints_steps": 50,
    "keep_checkpoint_max": 5,
    "save_summary_steps": 25,
    "log_step_count_steps": 10,

    # dataset - details
    "examples.conll2003.conll_2003_dataset.CoNLL2003Dataset": {
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
    "vitaflow.data.text.iterators.CSVSeqToSeqIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "conll_data_iterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "text_col": "0",
        "entity_col": "3",
        "batch_size": 4,
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
        "word_level_lstm_hidden_size": 12,
        "char_level_lstm_hidden_size": 12,
        "word_emd_size": 8,
        "char_emd_size": 8,
        "num_lstm_layers": 1,
        "keep_probability": 0.5,
    }
}
