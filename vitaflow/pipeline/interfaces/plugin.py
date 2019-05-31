import os
from abc import ABC
from glob import glob

from vitaflow.annotate_server.common import verify_isimagefile, verify_isfile

try:
    from vitaflow.annotate_server import config
    import sys
    # Add the draft folder path to the sys.path list
    sys.path.append('..')
except ImportError:
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


class ImagePluginAppModel(ABC):
    """Simple image processing plugin application

    Must implement `run` method for using."""

    def __init__(self,
                 input_file_types=['.JPG', '.jpg', '.png']):
        self._input_files_types = input_file_types
        self.source_dir = None
        self.destination_folder = None

    def _image_search(self, path):
        """Inputs should be collected as a list of tuples.
        """
        return find_files_with_ext(path, self._input_files_types)

    @property
    def is_in_memory(self):
        """Tells whether this plugin module can be used to chain for in memory processing """
        return False

    def get_all_source_files(self, source_dir):
        """Get the list of images files from the source directory"""
        return self._image_search(source_dir)

    def _handle_data(self, in_file_data):
        """Each plugin module should implement this to handle image array data"""
        raise NotImplementedError

    def _handle_file(self, in_file_path, out_file_path):
        raise NotImplementedError

    def _handle_files(self, source_dir, destination_dir):
        """Each plugin module should implement this to handle all the files in the given directory"""
        raise NotImplementedError

    def process_data(self, in_file_data):
        """Process the incoming image array data"""
        if not self.is_in_memory:
            return None
        return self._handle_data(in_file_data=in_file_data)

    def process_file(self, in_file_path, out_file_path):
        self._handle_file(in_file_path=in_file_path, out_file_path=out_file_path)

    def process_files(self,
                      source_dir,
                      destination_dir):
        """Process all the image files at the source location and store them in the destination directory"""
        # self._validate_inputs()
        self._handle_files(source_dir=source_dir, destination_dir=destination_dir)


class TextExtImagePluginModel(ImagePluginAppModel):
    """OCR Abstract Class"""

    def __init__(self):
        super().__init__()
        self.parallel_operator_func = None

    def bulk_run(self):
        if (not self.source_dir) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        all_images = self._image_search(self.source_dir + '/*')
        self.parallel_operator_func(all_images)


class StitchTextExtImagePluginModel(ImagePluginAppModel):
    """Abstract Class"""

    def __init__(self):
        super().__init__()
        self.parallel_operator_func = None

    def bulk_run(self):
        if (not self.source_dir) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        self.parallel_operator_func(os.path.join(self.root_folder, self.source_dir))
