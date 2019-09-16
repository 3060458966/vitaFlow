"""
``` bash
cd /path/to/vitaflow/
python vitaflow/bin/vf_engine.py --config_python_file=examples/cifar10/gan_config.py --mode=run_iterator
python vitaflow/bin/vf_engine.py --config_python_file=examples/cifar10/gan_config.py --mode=train
```
"""
import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "cfiar10_dataset"
batch_size = 32
noise_vec_dim = 30
experiments = {
    "num_epochs": 5,
    "dataset_class_with_path": "examples.cifar10.cifar10_dataset.Cifar10Dataset",
    "iterator_class_with_path": "examples.cifar10.cifar10_gan_iterator.Cifar10GanIterator",
    "model_class_with_path": "vitaflow.models.image.gan.began.BEGAN",
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
    "examples.cifar10.cifar10_gan_iterator.Cifar10GanIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "batch_size": batch_size,
        "noise_dim" : noise_vec_dim
    },


    # data model
    "vitaflow.models.image.gan.vanilla_gan.VanillaGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "gen_filter_size" : 512,
        "learning_rate" : 0.001,
        "alpha" : 0.15,
        "beta1" : 0.4,
        "z_dim" : noise_vec_dim
    },

    "vitaflow.models.image.gan.cgan.CGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "gen_filter_size" : 1024,
        "learning_rate" : 0.001,
        "alpha" : 0.15,
        "beta1" : 0.4,
        "z_dim" : noise_vec_dim
    },

    "vitaflow.models.image.gan.began.BEGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "gen_filter_size" : 1024,
        "learning_rate" : 0.001,
        "gamma" : 0.75,
        "lamda" : 0.001,
        "alpha" : 0.15,
        "beta1" : 0.4,
        "z_dim" : noise_vec_dim
    },
}

