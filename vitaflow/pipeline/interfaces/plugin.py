import os
from abc import ABC
from glob import glob
import concurrent.futures
from vitaflow.pipeline.interfaces import utils


try:
    from vitaflow.annotate_server import config
    import sys
    # Add the draft folder path to the sys.path list
    sys.path.append('..')
except ImportError:
    import config



def verify_isfile(full_path_file_name):
    return os.path.isfile(full_path_file_name)


def verify_isimagefile(full_path_file_name, exts=['.JPG', '.jpg', '.png']):
    return utils.get_file_ext(full_path_file_name) in exts



def find_files_with_ext(search_folder, exts=['.JPG', '.jpg', '.png']):
    all_files = glob(search_folder + '**/**', recursive=True)
    bag = []
    if exts:
        for _ext in exts:
            bag += [file for file in all_files if file.endswith(_ext)]
    else:
        bag = all_files
    return bag


class ImagePluginInterface(ABC):
    """Simple image processing plugin application

    Must implement `run` method for using."""

    def __init__(self):
        self.source_dir = None
        self.destination_folder = None

    @property
    def is_in_memory(self):
        """Tells whether this plugin module can be used to chain for in memory processing """
        return False

    @staticmethod
    def get_all_input_files(source_dir, input_files_types=['.JPG', '.jpg', '.png']):
        """Get the list of images files from the source directory"""
        return find_files_with_ext(source_dir, input_files_types)

    def _handle_data(self, in_file_data):
        """Plugin module should implement this to handle image array data"""
        raise NotImplementedError

    def _handle_file(self, in_file_path, out_file_path):
        raise NotImplementedError

    def _handle_files(self, source_dir, destination_dir):
        """Plugin module should implement this to handle all the files in the given directory"""

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        in_files = self.get_all_input_files(source_dir=source_dir)
        for img_file in in_files:
            filename = os.path.basename(img_file)
            in_file_path = img_file
            out_file_path = os.path.join(destination_dir, filename)
            self._handle_file(in_file_path=in_file_path, out_file_path=out_file_path)

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


class OCRPluginInterface(ImagePluginInterface):
    """OCR Abstract Class"""

    def __init__(self,
                 num_workers=4):
        ImagePluginInterface.__init__(self)
        self._num_workers = num_workers

    def _parallel(self, image_list, out_file_list):
        completed_jobs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            for img_path, out_file in zip(image_list, executor.map(self.process_file, image_list, out_file_list)):
                completed_jobs.append(
                    (img_path.split("\\")[-1], ',', out_file, ', processed')
                )

    def _handle_files(self, source_dir, destination_dir):
        """Plugin module should implement this to handle all the files in the given directory"""

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        in_files = self.get_all_input_files(source_dir=source_dir)
        out_files = []

        for img_file in in_files:
            filename = os.path.basename(img_file)
            dir_name = os.path.dirname(img_file).split("/")[-1]
            out_file_dir = os.path.join(destination_dir, dir_name)

            if not os.path.exists(out_file_dir):
                os.makedirs(out_file_dir)

            out_file_path = os.path.join(out_file_dir, filename + "_" + self.__class__.__name__ + "_.txt")
            out_files.append(out_file_path)

        self._parallel(image_list=in_files, out_file_list=out_files)


class TextCombiner(ImagePluginInterface):
    """Abstract Class"""

    def __init__(self):
        ImagePluginInterface.__init__(self)

