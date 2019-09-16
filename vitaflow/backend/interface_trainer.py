from vitaflow.datasets.interface_dataset import IDataset


class TrainerBase(object):
    def __init__(self,
                 experiment_name,
                 max_train_steps,
                 validation_interval_steps,
                 stored_model):
        self._experiment_name = experiment_name
        self._max_train_steps = max_train_steps
        self._validation_interval_steps = validation_interval_steps
        self._stored_model = stored_model

    def train(self, max_steps=None, num_epochs=None):
        raise NotImplementedError

    def evaluate(self, steps=None, checkpoint_path=None):
        raise NotImplementedError

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None, num_epochs=None):
        raise NotImplementedError

    def predict_directory(self, in_path, out_path):
        raise NotImplementedError

    def predict_file(self, in_path, out_path):
        raise NotImplementedError
