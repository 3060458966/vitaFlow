import gin
from vitaflow.datasets.image.icdar.icdar_data import ICDARTFDataset
from vitaflow.iterators.image.icdar_iterator import CIDARIterator
from vitaflow.models.image.east.east_model_v0 import EASTIEstimatorModel

@gin.configurable
def get_experiment_root_directory(value):
    return value
