import os
from abc import ABC
from glob import glob

from common import verify_isimagefile, verify_isfile
from vitaflow.annotate_server import config


def find_files_with_ext(search_folder, exts=None):
    all_files = glob(search_folder + '/*', recursive=True)
    bag = []
    if exts:
        for _ext in exts:
            bag += [file for file in all_files if file.endswith(_ext)]
    else:
        bag = all_files
    return bag


class PluginAppModel(ABC):
    '''Simple image processing plugin application

    Must implement `run` method for using.'''

    def __init__(self):
        self._inputs = None
        self._inputs_validated = False
        self._destination_out_exists = False
        # custom location according to need
        self.source_folder = None
        self.destination_folder = None
        # transformation function for converting source_image to destination_image
        self.operator_func = None

    def plugin_inputs(self):
        # custom location according to need
        self.source_folder = None
        self.destination_folder = None
        # transformation function for converting source_image to destination_image
        self.operator_func = None

    @staticmethod
    def image_search(path):
        """Inputs should be collected as a list of tuples.

        Each tuple items shall contains args to pass to input method
        """
        path = os.path.join(config.ROOT_DIR, path)
        return find_files_with_ext(path, config.IMAGE_EXTS)

    def inputs(self, source_image, destination_image=None):
        """Validate the inputs"""
        # verify source
        if not all([
            verify_isfile(source_image),
            verify_isimagefile(source_image)
        ]):
            print([
                verify_isfile(source_image),
                verify_isimagefile(source_image)
            ])
            raise ValueError('inputs are ')

        # set destination
        if destination_image is None:
            if not self.destination_folder:
                raise AttributeError('self.destination_folder is not set !!')
            filename = os.path.basename(source_image)
            destination_image = os.path.join(config.ROOT_DIR, self.destination_folder, filename)

        # for customisation - write code here
        self._inputs = (source_image, destination_image)
        self._inputs_validated = True
        self._destination_out_exists = verify_isfile(destination_image)

    def _validate_inputs(self):
        if not self._inputs_validated:
            raise ValueError('Input Validations is inComplete')

    def run(self):
        """Execute of Code logic"""
        self._validate_inputs()
        if self._destination_out_exists:
            print('Destination{} already skipping run'.format(self._inputs[-1]))
        try:
            if self.operator_func:
                self.operator_func(*self._inputs)
            else:
                raise RuntimeError('self.operator_func is not defined !!')
        except:
            raise RuntimeError('self.operator_func failed !!')

    def quick_run(self, *args):
        """For controlled or sequential runs"""
        self.inputs(*args)
        self.run()

    def bulk_run(self):
        """For automated runs"""
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        all_source_images = self.image_search(self.source_folder)
        for source_image in all_source_images:
            self.quick_run(source_image, None)


class TextExtPluginModel(PluginAppModel):
    '''OCR Abstract Class'''

    def __init__(self):
        super().__init__()
        self.parallel_operator_func = None

    def bulk_run(self):
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        all_images = self.image_search(self.source_folder + '/*')
        self.parallel_operator_func(all_images)
