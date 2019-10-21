import os

FRAMES_PER_SAMPLE = 100  # number of frames forming a chunk of data
SAMPLING_RATE = 16000
FRAME_SIZE = 256
NEFF = 129  # effective FFT points
# amplification factor of the waveform sig
AMP_FAC = 10000 # i.e log(10000) = 4
MIN_AMP = 10000
# TF bins smaller than THRESHOLD will be
# considered inactive
THRESHOLD = 40
# prams for pre-whitening
GLOBAL_MEAN = 44
GLOBAL_STD = 15.5
LEARNING_RATE = 1e-2

experiment_root_directory = "/opt/data/vitaflow/"
experiment_name = "TEDLiumDataset"
batch_size = 96

experiments = {
    "num_epochs": 20,
    "dataset_class_with_path": "vitaflow.playground.shabda.tedlium_dataset.TEDLiumDataset",
    "iterator_class_with_path": "vitaflow.playground.shabda.tedlium_parallel_iterator_tfrecord.TEDLiumIterator",
    "model_class_with_path": "vitaflow.playground.shabda.deep_clustering.DeepClustering",
    "store_model_interval_steps": 100,
    "keep_checkpoint_max": 5,
    "save_summary_steps": 10,
    "log_step_count_steps": 10,
    "clear_model_data" : False,
    "plug_dataset" : True,


    "vitaflow.playground.shabda.tedlium_dataset.TEDLiumDataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "train_data_path": "train",
        "validation_data_path": "dev",
        "test_data_path": "test",
        "num_clips": 25,
        "duration": 30,
        "sampling_rate" : SAMPLING_RATE,
        "frame_size" : FRAME_SIZE,
        "neff" : NEFF,
        "min_amp" : MIN_AMP,
        "amp_fac" : AMP_FAC,
        "threshold" : THRESHOLD,
        "global_mean" : GLOBAL_MEAN,
        "global_std" : GLOBAL_STD,
        "frames_per_sample" : FRAMES_PER_SAMPLE,
        "reinit_file_pair" : False,
        "spark_master" : "local[4]",
    },


    "vitaflow.playground.shabda.tedlium_parallel_iterator_tfrecord.TEDLiumIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "dev",
        "test_data_path": "test",
        "sampling_rate" : SAMPLING_RATE,
        "frame_size" : FRAME_SIZE,
        "neff" : NEFF,
        "min_amp" : MIN_AMP,
        "amp_fac" : AMP_FAC,
        "threshold" : THRESHOLD,
        "global_mean" : GLOBAL_MEAN,
        "global_std" : GLOBAL_STD,
        "frames_per_sample" : FRAMES_PER_SAMPLE,
        "batch_size" : batch_size,
        "prefetch_size" : 1,
        "num_threads" : 3,

    },

    "vitaflow.playground.shabda.deep_clustering.DeepClustering" : {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "neff" : NEFF,
        "batch_size" : batch_size,
        "lstm_hidden_size" : 512,
        "p_keep_ff" : 0.9,
        "p_keep_rc" : 0.8,
        "frames_per_sample" : FRAMES_PER_SAMPLE,
        "embd_dim_k" : 60, # 5 to 60
        "learning_rate" : LEARNING_RATE
    }
}
