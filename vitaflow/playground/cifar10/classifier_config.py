"""
``` bash
cd /path/to/vitaflow/
python vitaflow/bin/run_experiments.py --config_python_file=examples/cifar10/classifier_config.py --mode=run_iterator
python vitaflow/bin/run_experiments.py --config_python_file=examples/cifar10/classifier_config.py --mode=train
```
"""
import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "cfiar10_dataset"
batch_size = 64

experiments = {
    "num_epochs": 100,
    "dataset_class_with_path": "examples.cifar10.cifar10_dataset.Cifar10Dataset",
    "iterator_class_with_path": "examples.cifar10.cifar10_basic_iterator.Cifar10BasicIterator",
    "model_class_with_path": "vitaflow.models.image.classification.naive_convnet.NaiveConvNet",
    "save_checkpoints_steps" : (45000 // batch_size) // 5,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : (45000 // batch_size) // 100,
    "log_step_count_steps" : 2,

    # dataset - details
    "examples.cifar10.cifar10_dataset.Cifar10Dataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "over_write": False,
    },

    # data iterator
    "examples.cifar10.cifar10_basic_iterator.Cifar10BasicIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "batch_size": batch_size,
    },

    # data model
    "vitaflow.models.image.classification.naive_convnet.NaiveConvNet" : {
        "modehparams.l_root_directory":  experiment_root_directory,
        "experiment_name": experiment_name,
        # hyper parameters
        "keep_probability": 0.5,
        "out_dim": 10,
        "learning_rate": 0.001,
    }

}

