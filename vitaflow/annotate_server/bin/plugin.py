import os
from abc import ABC
from glob import glob

from common import verify_isimagefile, verify_isfile

try:
    from vitaflow.annotate_server import config
    import sys

    # Add the draft folder path to the sys.path list
    sys.path.append('..')
except ModuleNotFoundError:
    import config


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
    """Simple image processing plugin application

    Must implement `run` method for using."""

    def __init__(self):
        self._inputs = None
        self._inputs_validated = False
        self._input_files_types = config.IMAGE_EXTS
        self._destination_out_exists = False
        # Custom location according to need #
        # main folders
        self.root_folder = config.ROOT_DIR
        # relative path
        self.source_folder = None
        # relative path
        self.destination_folder = None
        # Transformation fns to convert source_file to dest_file #
        self.operator_func = None

    # TODO: remove this fns & re-write it  dependencies
    def plugin_inputs(self):
        # custom location according to need
        self.source_folder = None
        self.destination_folder = None
        # transformation function for converting source_file to destination_file
        self.operator_func = None

    def image_search(self, path):
        """Inputs should be collected as a list of tuples.

        Each tuple items shall contains args to pass to input method
        """
        path = os.path.join(self.root_folder, path)
        return find_files_with_ext(path, self._input_files_types)

    # @staticmethod
    # def pdf_search(path):
    #     """Inputs should be collected as a list of tuples.
    #
    #     Each tuple items shall contains args to pass to input method
    #     """
    #     path = os.path.join(self.root_folder, path)
    #     return find_files_with_ext(path, config.PDF_EXTS)

    def inputs(self, source_file, destination_file=None):
        """Validate the inputs"""
        # Verify source
        if not all([
            verify_isfile(source_file),
            verify_isimagefile(source_file)
        ]):
            print([
                verify_isfile(source_file),
                verify_isimagefile(source_file)
            ])
            raise ValueError('inputs are ')

        # set destination
        if destination_file is None:
            if not self.destination_folder:
                raise AttributeError('self.destination_folder is not set !!')
            filename = os.path.basename(source_file)
            destination_file = os.path.join(self.root_folder, self.destination_folder, filename)

        # for customisation - write code here
        self._inputs = (source_file, destination_file)
        self._inputs_validated = True
        self._destination_out_exists = verify_isfile(destination_file)

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
        except Exception as ex:
            print(ex)
            raise RuntimeError('self.operator_func failed !!')

    def quick_run(self, *args):
        """For controlled or sequential runs"""
        self.inputs(*args)
        self.run()

    def bulk_run(self):
        """For automated runs"""
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        all_source_files = self.image_search(self.source_folder)
        for source_file in all_source_files:
            self.quick_run(source_file, None)


class TextExtPluginModel(PluginAppModel):
    """OCR Abstract Class"""

    def __init__(self):
        super().__init__()
        self.parallel_operator_func = None

    def bulk_run(self):
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        all_images = self.image_search(self.source_folder + '/*')
        self.parallel_operator_func(all_images)


class StitchTextExtPluginModel(PluginAppModel):
    """Abstract Class"""

    def __init__(self):
        super().__init__()
        self.parallel_operator_func = None

    def bulk_run(self):
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        self.parallel_operator_func(os.path.join(self.root_folder, self.source_folder))
