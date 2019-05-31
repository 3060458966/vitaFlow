import os
import subprocess

from vitaflow.pipeline.interfaces.plugin import ImagePluginAppModel

_command_convert = ['convert',
                    '-auto-level',
                    '-sharpen',
                    '0x4.0',
                    '-contrast']

_command_textcleaner = ['./textcleaner',
                        '-g',
                        '-e',
                        'stretch',
                        '-f',
                        '25',
                        '-o',
                        '10',
                        '-u',
                        '-s',
                        '1',
                        '-T',
                        '-p',
                        '10']

# TODO: need to optimise
#  http://www.fmwconcepts.com/imagemagick/color2gray/index.php
_command_color2gray = ['convert',
                       '-colorspace',
                       'Gray']

# TODO: need to optimise
# http://www.fmwconcepts.com/imagemagick/graytoning/index.php
_command_graytoning = ['./graytoning',
                       '-r 60',
                       '-g 20',
                       '-b 20']

def _color2gray(image_loc, dest_image_loc):
    subprocess.check_call(_command_color2gray + [image_loc, dest_image_loc])


def _graytoning(image_loc, dest_image_loc):
    subprocess.check_call(_command_graytoning + [image_loc, dest_image_loc])


def _convert(image_loc, dest_image_loc):
    subprocess.check_call(_command_convert + [image_loc, dest_image_loc])


def _binarisation(image_loc, dest_image_loc):
    subprocess.check_call(_command_textcleaner + [dest_image_loc, dest_image_loc])


def binarisation(image_loc, dest_image_loc):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    _convert(image_loc, dest_image_loc)
    # _convert(dest_image_loc, dest_image_loc)
    _color2gray(dest_image_loc, dest_image_loc)


# noinspection PyUnusedLocal
def blur(image_loc, dest_image_loc):
    pass


class ImageBinarisePreprocessor(ImagePluginAppModel):
    def __init__(self,
                 root_dir=None,
                 source_folder=None,
                 destination_folder=None):
        ImagePluginAppModel.__init__(self)

    # def _validate_inputs(self):
    #     if not self._inputs_validated:
    #         raise ValueError('Input Validations is inComplete')

    def _handle_data(self, in_file_data):
        """Each plugin module should implement this to handle image array data"""
        raise NotImplementedError

    def _handle_file(self, in_file_path, out_file_path):
        if os.path.isfile(out_file_path):
            print('Binarisation found existing file {}'.format(out_file_path))
            return
        try:
            binarisation(in_file_path, out_file_path)
            print('Binarisation generated file {}'.format(out_file_path))
        except Exception as e:
            print(e)
            print('Binarisation - Failed - Generated file {}'.format(out_file_path))

    def _handle_files(self, source_dir, destination_dir):
        """Each plugin module should implement this to handle all the files in the given directory"""

        print('binarisation src {} dest {} '.format(source_dir, destination_dir))
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        in_files = self.get_all_source_files(source_dir=source_dir)
        print(in_files)
        for img_file in in_files:
            filename = os.path.basename(img_file)
            in_file_path = img_file
            out_file_path = os.path.join(destination_dir, filename)
            self._handle_file(in_file_path=in_file_path, out_file_path=out_file_path)

if __name__ == '__main__':
    t = ImageBinarisePreprocessor()
    print('--' * 55)
    t.process_files(source_dir="/opt/tmp/vitaFlow/east/", destination_dir="/opt/tmp/vitaFlow/east_binarized/")
