import os
import subprocess

import config
from bin.plugin import PluginAppModel

_command_convert = ['/usr/local/bin/convert',
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


def _convert(image_loc, dest_image_loc):
    subprocess.check_call(_command_convert + [image_loc, dest_image_loc])


def _binarisation(image_loc, dest_image_loc):
    subprocess.check_call(_command_textcleaner + [dest_image_loc, dest_image_loc])


def binarisation(image_loc, dest_image_loc):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    _convert(image_loc, dest_image_loc)
    _convert(dest_image_loc, dest_image_loc)


# noinspection PyUnusedLocal
def blur(image_loc, dest_image_loc):
    pass


def main(image_loc, dest_image_loc=None):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    if dest_image_loc is None:
        filename = os.path.basename(image_loc)
        dest_image_loc = os.path.join(config.ROOT_DIR, config.BINARIZE_ROOT_DIR, filename)

    if os.path.isfile(dest_image_loc):
        print('Binarisation found existing file {}'.format(dest_image_loc))
        return
    try:
        binarisation(image_loc, dest_image_loc)
        print('Binarisation generated file {}'.format(dest_image_loc))
    except:
        print('Binarisation - Failed - Generated file {}'.format(dest_image_loc))


class imageBinarisePlugin(PluginAppModel):

    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.IMAGE_ROOT_DIR
        self.destination_folder = config.BINARIZE_ROOT_DIR
        # Transformation function for converting source_image to destination_image
        self.operator_func = main


if __name__ == '__main__':
    t = imageBinarisePlugin()
    t.plugin_inputs()
    print('--' * 55)
    t.bulk_run()
