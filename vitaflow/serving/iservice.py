import os
from vitaflow.models.interface_model import IEstimatorModel, ITorchModel, IKerasModel
from vitaflow.backend.torch_trainer import TorchTrainer
from vitaflow.backend.keras_trainer import KerasTrainer
from vitaflow.backend.tf_executor import TFExecutor
from vitaflow.datasets.datasets import get_dataset
from vitaflow.models.models import get_model

class IServing():
    def __init__(self,
                 dataset_name,
                 model_name,
                 model_store_path):
        self._dataset = get_dataset(dataset_name)
        self._model = get_model(model_name)

        # Initialize the dataset and _model
        self._dataset = self._dataset()
        self._model = self._model(dataset=self._dataset)

        if isinstance(self._model, ITorchModel):
            self._executor = TorchTrainer(model=self._model,
                                          dataset=self._dataset,
                                          model_store_path=model_store_path)


        # elif isinstance(self._model, IEstimatorModel):
        #     self._init_tf_config()
        #     executor = TFExecutor(model=self._model,
        #                           dataset=self._dataset,
        #                           config=self._run_config,
        #                           max_steps_without_decrease=self.max_steps_without_decrease)
        #     memory_usage_psutil()
        #     executor.train(num_epochs=self._num_epochs)
        #     executor.evaluate(steps=None)
        #
        # elif isinstance(self._model, IKerasModel):
        #     trainier = KerasTrainer(model=self._model,
        #                             dataset=self._dataset,
        #                             model_store_path="",
        #                             nb_workers=4)
        #     trainier.train_and_evaluate(max_train_steps=None, eval_steps=None, num_epochs=self._num_epochs)

    def get_metrics(self, test_dir):
        raise NotImplementedError

    def predict(self,
                in_file_or_path,
                out_file_or_path):
        """
        Runs model prediction
        :param in_file_or_path: Input file path or folder path
        :param out_file_or_path: Output file path or folder path
        :return:
        """
        raise NotImplementedError



