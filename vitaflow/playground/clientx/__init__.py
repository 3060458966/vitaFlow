import gin

from vitaflow.playground.clientx.clientx_dataset import CLIENTXDataset


@gin.configurable
def get_experiment_root_directory(value):
    return value
