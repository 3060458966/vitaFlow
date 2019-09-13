import os
import subprocess
from PIL import Image
from keras.optimizers import Adam

from vitaflow.pipeline.interfaces.plugin import ImagePluginInterface
from vitaflow import demo_config
from vitaflow.pipeline.preprocessor.nnet_binarizer import unet, nnet_binarizes

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


class ImageBinarisePreprocessor(ImagePluginInterface):
    def __init__(self,
                 weights_path):
        ImagePluginInterface.__init__(self)
        self._weights_path = weights_path

        # self._model = unet()
        # self._model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        # self._model.load_weights(self._weights_path)

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
        # try:
        #     im = Image.open(in_file_path)
        #     dpi = 300#im.info['dpi']
        #     if dpi is not None and dpi > 299:
        #         nnet_binarizes(in_file_path, out_file_path, model=self._model)
        #     else:
        #         binarisation(in_file_path, out_file_path)
        #     print('Binarisation generated file {}'.format(out_file_path))
        # except Exception as e:
        #     print(e)
        #     print('Binarisation - Failed - Generated file {}'.format(out_file_path))
        #

if __name__ == '__main__':
    t = ImageBinarisePreprocessor(weights_path=None) #TODO fix
    print('--' * 55)
    t.process_files(source_dir=demo_config.EAST_OUT_DIR, destination_dir=demo_config.BINARIZE_ROOT_DIR)
