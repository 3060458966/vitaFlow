import gin
from vitaflow.serving.iservice import IServing
from vitaflow.models.image.east.east_torch_detect import *
from vitaflow.utils.registry import register_serving

@gin.configurable
@register_serving
class EastTorchServing(IServing):

    def __init__(self,
                 dataset_name,
                 model_name,
                 model_store_path):
        IServing.__init__(self, dataset_name=dataset_name,
                          model_name=model_name,
                          model_store_path=model_store_path)

    def predict(self,
                file_path,
                out_file_path=None):
        img = Image.open(file_path)
        boxes = detect(img, self._model.module, self._executor.device)
        plot_img = plot_boxes(img, boxes)
        plot_img.save(out_file_path)



