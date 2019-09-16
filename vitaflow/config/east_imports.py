import gin

from vitaflow.datasets.image.icdar.icdar_data import CDARDataset
# from vitaflow.models.image.east.east_model_v0 import EASTIEstimatorModel
from vitaflow.models.image.east.east_model_keras_v2 import EASTV2Keras


@gin.configurable
def get_experiment_root_directory(value):
    return value
