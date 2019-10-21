import os
import sys
import time
from abc import abstractmethod
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from vitaflow.datasets.image.scene_text_recognition.utils import Averager
from vitaflow.backend.interface_trainer import TrainerBase
from vitaflow.models.interface_model import ITorchModel
from vitaflow.utils.print_helper import print_info, print_warn
from vitaflow.utils.torch.visualization import TensorboardWriter
from vitaflow.models.image.east.east_torch_detect import detect


class TorchTrainer(TrainerBase):

    def __init__(self,
                 model,
                 dataset,
                 model_store_path,
                 num_gpu=1):

        TrainerBase.__init__(self,
                             model=model,
                             dataset=dataset,
                             model_store_path=model_store_path)


        # setup GPU device if available, move _model into configured device
        self.device, self.device_ids = self._prepare_device(num_gpu)

        self._model.set_device(self.device)
        self._module = model.module.to(self.device)

        latest_weights_path = os.path.join(self._model_store_path, "latest_weights.pth")
        self._module = self._resume_checkpoint(model_store_path=latest_weights_path, module=self._module)

        self._data_parallel = False
        if len(self.device_ids) > 1:
            self._module = torch.nn.DataParallel(self._module, device_ids=self.device_ids)
            self._data_parallel = True


        if not os.path.exists(self._model_store_path):
            os.makedirs(self._model_store_path)

        print_info("Model weights will be stored in following path : {}".format(self._model_store_path))

    def train(self,
              num_max_steps=None,
              num_epochs=None,
              store_model_epoch_interval=None,
              store_model_steps_interval=None):
        """
        Full training logic
        """
        for epoch in range(1, num_epochs+1):
            current_epoch_model_store_path = os.path.join(self._model_store_path, 'epoch_{}.pth'.format(epoch))
            previous_epoch_model_store_path = None
            if os.path.exists(current_epoch_model_store_path):
                print_info("Found : {}".format(current_epoch_model_store_path))
                previous_epoch_model_store_path = os.path.join(self._model_store_path, 'epoch_{}.pth'.format(epoch - 1))
                continue
            else:
                self._module = self._resume_checkpoint(module=self._module,
                                                       model_store_path=previous_epoch_model_store_path)
                self._model.load_module(module=self._module)
                self._model._train_epoch(epoch)

            if epoch % store_model_epoch_interval == 0:
                self._save_checkpoint(epoch=epoch, model_store_path=self._model_store_path)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move _model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print_warn("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print_warn("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, model_store_path, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        state_dict = self._module.module.state_dict() if self._data_parallel else self._module.state_dict()
        torch.save(state_dict, os.path.join(model_store_path, 'epoch_{}.pth'.format(epoch)))
        #For eacy loading of last stored weights
        torch.save(state_dict, os.path.join(model_store_path, 'latest_weights.pth'.format(epoch)))

    def _resume_checkpoint(self, module, model_store_path):
        """
        Resume from saved checkpoints
        :param model:
        :param model_store_path:
        :return:
        """
        if model_store_path is None:
            return module
        else:
            if os.path.isfile(model_store_path) and os.path.exists(model_store_path):
                print_info(f'loading pretrained _model from {model_store_path}')
                module.load_state_dict(torch.load(model_store_path))
                module.eval()
        return module

    def predict(self, x):
        return self._module(x)