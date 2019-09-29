import gin

from vitaflow.datasets.image.scene_text_recognition.str_dataset import SceneTextRecognitionDataset
from vitaflow.models.image.str.str_models import SceneTextRecognitionModel

@gin.configurable
def get_experiment_root_directory(value):
    return value
