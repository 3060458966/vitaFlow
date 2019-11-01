import os
import time

import gin
import string
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.metrics.distance import edit_distance

from vitaflow.utils.print_helper import print_info
from vitaflow.utils.registry import register_model

import numpy as np

from vitaflow.models.interface_model import ITorchModel
from vitaflow.models.image.str.modules.transformation import TPS_SpatialTransformerNetwork
from vitaflow.models.image.str.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from vitaflow.models.image.str.modules.sequence_modeling import BidirectionalLSTM
from vitaflow.models.image.str.modules.prediction import Attention
from vitaflow.datasets.image.scene_text_recognition.utils import AttnLabelConverter, Averager, CTCLabelConverter


class SceneTextRecognitionModule(nn.Module):

    def __init__(self,
                 batch_max_length,
                 transformation_stage="TPS",
                 feature_extraction_stage="ResNet",
                 sequence_modeling_stage="BiLSTM",
                 prediction_stage="Attn",
                 img_width=32,
                 img_height=100,
                 input_channel=1,
                 output_channel=512,
                 num_fiducial=20,
                 hidden_size=256,
                 is_adam=True,
                 character=None,
                 is_sensitive=True):
        super(SceneTextRecognitionModule, self).__init__()
        self.stages = {"transformation_stage": transformation_stage,
                       "feature_extraction_stage": feature_extraction_stage,
                       "sequence_modeling_stage": sequence_modeling_stage,
                       "prediction_stage": prediction_stage}

        self.batch_max_length = batch_max_length

        self.is_adam = is_adam
        self.character = character

        if is_sensitive:
            # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.character = string.printable[:-6]

        self.num_classes = len(self.character)

        """ Transformation """
        if transformation_stage == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial,
                I_size=(img_height, img_width),
                I_r_size=(img_height, img_width),
                I_channel_num=input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if feature_extraction_stage == 'VGG':
            self.feature_extraction_model = VGG_FeatureExtractor(input_channel, output_channel)
        elif feature_extraction_stage == 'RCNN':
            self.feature_extraction_model = RCNN_FeatureExtractor(input_channel, output_channel)
        elif feature_extraction_stage == 'ResNet':
            self.feature_extraction_model = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')

        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512

        self.adaptive_avg_pool_layer = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if sequence_modeling_stage == 'BiLSTM':
            self.sequence_model = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if prediction_stage == 'CTC':
            self.prediction_model = nn.Linear(self.SequenceModeling_output, self.num_classes)
        elif prediction_stage == 'Attn':
            self.prediction_model = Attention(self.SequenceModeling_output, hidden_size, self.num_classes)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages["transformation_stage"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.feature_extraction_model(input)
        visual_feature = self.adaptive_avg_pool_layer(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["sequence_modeling_stage"] == 'BiLSTM':
            contextual_feature = self.sequence_model(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["prediction_stage"] == 'CTC':
            prediction = self.prediction_model(contextual_feature.contiguous())
        else:
            prediction = self.prediction_model(contextual_feature.contiguous(),
                                               text,
                                               is_train,
                                               batch_max_length=self.batch_max_length)

        return prediction



@gin.configurable
@register_model
class SceneTextRecognitionModel(ITorchModel):

    def __init__(self,
                 dataset,
                 learning_rate=0.01,
                 num_epochs=5,
                 model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "str_model"),
                 transformation_stage="TPS",
                 feature_extraction_stage="ResNet",
                 sequence_modeling_stage="BiLSTM",
                 prediction_stage="Attn",
                 img_width=32,
                 img_height=100,
                 input_channel=1,
                 output_channel=512,
                 num_fiducial=20,
                 hidden_size=256,
                 is_adam=True,
                 character=None,
                 is_sensitive=True,
                 batch_size=192,
                 batch_max_length=25):
        super(SceneTextRecognitionModel, self).__init__(model_root_directory=model_root_directory,
                                                        dataset=dataset,
                                                        learning_rate=learning_rate,
                                                        module=SceneTextRecognitionModule(batch_max_length=batch_max_length,
                                                                                          transformation_stage=transformation_stage,
                                                                                          feature_extraction_stage=feature_extraction_stage,
                                                                                          sequence_modeling_stage=sequence_modeling_stage,
                                                                                          prediction_stage=prediction_stage,
                                                                                          img_width=img_width,
                                                                                          img_height=img_height,
                                                                                          input_channel=input_channel,
                                                                                          output_channel=output_channel,
                                                                                          num_fiducial=num_fiducial,
                                                                                          hidden_size=hidden_size,
                                                                                          is_adam=is_adam,
                                                                                          character=character,
                                                                                          is_sensitive=is_sensitive))
        self.prediction_stage = prediction_stage
        self.stages = {"transformation_stage": transformation_stage,
                       "feature_extraction_stage": feature_extraction_stage,
                       "sequence_modeling_stage": sequence_modeling_stage,
                       "prediction_stage": prediction_stage}

        self.is_adam = is_adam
        self.character = character
        self.batch_size = batch_size
        self.batch_max_length = batch_max_length
        self._model_root_directory = model_root_directory

        self.grad_clip = 5 #gradient clipping value. default=5

        if is_sensitive:
            # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.character = string.printable[:-6]

    @property
    def name(self):
        return "scene_text_recognition_dataset"

    @property
    def model_dir(self):
        """
        Returns _model directory `model_root_directory`/SceneTextRecognitionModel
        :return:
        """
        return os.path.join(self._model_root_directory,
                            type(self).__name__)

    def compile(self, *args, **kargs):
        self._criterion = self.get_loss_op()
        self._optimizer = self.get_optimizer()
        # self._scheduler = lr_scheduler.MultiStepLR(self._optimizer, milestones=[self._num_epochs // 2], gamma=0.1)
        # self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #TODO
        self._converter = self.get_converter()


    def get_loss_op(self):
        """ setup loss """
        if 'CTC' in self.stages["prediction_stage"]:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(self._device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self._device)  # ignore [GO] token = ignore index 0

        return criterion

    def get_cost(self, model, features, labels):
        assert (isinstance(model, SceneTextRecognitionModel))

        converter = model.get_converter()

        text, length = converter.encode(labels, batch_max_length=25) #self.batch_max_length)
        batch_size = features.size(0)

        criterion = self.get_loss_op()

        image, text, length = features, text, length
        if 'CTC' in self.stages["prediction_stage"]:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(self._device)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text, preds_size, length)
            torch.backends.cudnn.enabled = True

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        return cost

    def get_converter(self):
        """ _model configuration """
        if 'CTC' in self.stages["prediction_stage"]:
            converter = CTCLabelConverter(self.character)
        else:
            converter = AttnLabelConverter(self.character)
        self.num_class = len(converter.character)

        return converter

    def get_optimizer(self):
        # assert (isinstance(_model, SceneTextRecognitionModel))
        # filter that only require gradient decent
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, self.module.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, _model.named_parameters())]

        # setup optimizer
        if self.is_adam:
            optimizer = optim.Adam(filtered_parameters, lr=1.0, betas=(0.9, 0.999))
        else:
            optimizer = optim.Adadelta(filtered_parameters, lr=1.0, rho=0.95, eps=1e-8)
        print("Optimizer:")
        print(optimizer)

        return optimizer

    def get_predictions(self, model, batch_size, images, labels=None):
        start_time = time.time()

        criterion = self.get_loss_op()
        batch_max_length = 25
        converter = self.get_converter()

        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(self._device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(self._device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=batch_max_length)

        if labels:

            if 'CTC' in self.stages["prediction_stage"]:
                preds = model(images, text_for_pred).log_softmax(2)
                forward_time = time.time() - start_time

                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)  # to use CTCloss format

                # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
                # https://github.com/jpuigcerver/PyLaia/issues/16
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text_for_loss, preds_size, length_for_loss)
                torch.backends.cudnn.enabled = True

                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(images, text_for_pred, is_train=False)
                forward_time = time.time() - start_time

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            return forward_time, cost, preds_str, labels

        else: #TODO clean later
            # For max length prediction
            # length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            # text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in self.stages["prediction_stage"]:
                preds = model(images, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(images, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_str_cleaned = []
            for pred in preds_str:
                if 'Attn' in self.stages["prediction_stage"]:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                    preds_str_cleaned.append(pred)

            return preds_str_cleaned

    def get_accuracy(self, features, labels):
        n_correct = 0
        norm_ED = 0
        # calculate accuracy.
        for pred, gt in zip(features, labels):
            if 'Attn' in self.stages["prediction_stage"]:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                gt = gt[:gt.find('[s]')]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

        return n_correct, norm_ED


    def _train_epoch(self, epoch):
        for data_loader in self._dataset.get_torch_train_data_loaders():
            for i, (image_tensors, labels) in enumerate(data_loader):
                # image_tensors, labels = self._dataset.get_torch_train_dataset().get_batch()
                image = image_tensors.to(self._device)
                text, length = self._converter.encode(labels, batch_max_length=self.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in self.stages["prediction_stage"]:
                    preds = self._module(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(self._device)
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format

                    # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
                    # https://github.com/jpuigcerver/PyLaia/issues/16
                    torch.backends.cudnn.enabled = False
                    cost = self._criterion(preds, text, preds_size, length)
                    torch.backends.cudnn.enabled = True

                else:
                    preds = self.module(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = self._criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

                self.module.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)  # gradient clipping with 5 (Default)
                self._optimizer.step()

                # loss_avg.add(cost)
                print_info("Batch {} over".format(i))

