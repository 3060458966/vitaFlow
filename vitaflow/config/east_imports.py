import gin

from vitaflow.datasets.image.icdar.icdar_data import ICDARDataset
# from vitaflow.models.image.east.east_model_v0 import EASTIEstimatorModel
from vitaflow.models.image.east.east_model_keras_v2 import EASTV2Keras
from vitaflow.models.image.east.east_model_v1 import EASTTFModel
from vitaflow.models.image.east.east_torch_model import EASTTorchModel
from vitaflow.datasets.image.icdar.icdar_dataset_v1 import ICDARDatasetV1
from vitaflow.serving.east_torch_serving import EastTorchServing

@gin.configurable
def get_experiment_root_directory(value):
    return value
