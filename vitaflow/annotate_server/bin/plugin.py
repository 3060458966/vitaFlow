from abc import ABC, abstractmethod


class pluginApplication(ABC):
    '''Simple image processing plugin application

    Must implement `run` method for using.'''

    def __init__(self):
        self._inputs = None
        self._inputs_validated = False

    @abstractmethod
    def inputs(self, *args):
        "Validate the inputs"
        self._inputs = args
        # write code here
        self._inputs_validated = True

    def validate_inputs(self):
        if not self._input_validated:
            raise ValueError('Input Validations is inComplete')

    @abstractmethod
    def run(self):
        "Execute of Code logic"
        self.validate_inputs()
        # write code here
        raise NotImplementedError('Plugin.run method is not implemented')

    def quick_run(self, *args):
        self.inputs(*args)
        self.run()


class textExtraction(pluginApplication):
    '''OCR Abstract Class'''

    def quick_run(self):
        self.inputs()
        self.run()
