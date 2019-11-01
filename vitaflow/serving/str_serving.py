import gin
import torch

from vitaflow.datasets.image.scene_text_recognition.str_dataset import SceneTextRecognitionDataset
from vitaflow.models.image.str.str_models import SceneTextRecognitionModel
from vitaflow.serving.iservice import IServing
from vitaflow.utils.registry import register_serving
from vitaflow.datasets.image.scene_text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from vitaflow.datasets.image.scene_text_recognition.str_dataset import RawDataset, AlignCollate

@gin.configurable
@register_serving
class StrServing(IServing):

    def __init__(self,
                 dataset_name,
                 model_name,
                 model_store_path):
        IServing.__init__(self, dataset_name=dataset_name,
                          model_name=model_name,
                          model_store_path=model_store_path)
        self._dataset: SceneTextRecognitionDataset = self._dataset
        self._model: SceneTextRecognitionModel = self._model

        self.prediction_stage = self._model.prediction_stage
        if 'CTC' in self.prediction_stage:
            self.converter = CTCLabelConverter(self._dataset._character)
        else:
            self.converter = AttnLabelConverter(self._dataset._character)
        self.num_class = len(self.converter.character)

        if self._dataset._is_rgb:
            self.input_channel = 3

        self.device = self._model._device

        self.module = self._model._module

        self.img_height = self._dataset._img_height,
        self.img_width = self._dataset._img_width
        self.is_pad = self._dataset._is_pad
        self.batch_size = self._dataset._batch_size
        self.num_cores = self._dataset._num_cores
        self.is_rgb = self._dataset._is_rgb
        self.batch_max_length = self._dataset._batch_max_length
        self.batch_size = self._dataset._batch_size

    def predict(self,
                in_file_or_path,
                out_file_or_path):

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        align_collate_demo = AlignCollate(img_height=self.img_height,
                                          img_width=self.img_width,
                                          keep_ratio_with_pad=self.is_pad)
        demo_data = RawDataset(in_file_or_path,
                               self.is_rgb,
                               self.img_height,
                               self.img_width)  # use RawDataset

        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.num_cores),
            collate_fn=align_collate_demo, pin_memory=True)

        # predict
        self.module.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self._model._device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.batch_max_length] * self.batch_size).to(self.device)
                text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.prediction_stage:
                    preds = self.module(image, text_for_pred).log_softmax(2)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * self.batch_size)
                    _, preds_index = preds.permute(1, 0, 2).max(2)
                    preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = self.module(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                print('-' * 80)
                print('image_path\tpredicted_labels')
                print('-' * 80)
                for img_name, pred in zip(image_path_list, preds_str):
                    if 'Attn' in self.prediction_stage:
                        pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                    print(f'{img_name}\t{pred}')




