import os
import subprocess

from vitaflow.pipeline.interfaces.plugin import ModuleInterface

_command_convert = ['/usr/bin/convert',
                    '-density', '300',
                    '-depth', '8',
                    '-flatten', '+matte'
                    ]

_command_tesseract = ['tesseract', '-c', 'perserve_interword_spaces=1',
                      '--psm', '1',
                      '--oem', '1',
                      'txt'
                      ]


def _convert(pdf_loc, dest_image_loc):
    subprocess.check_call(_command_convert + [pdf_loc, dest_image_loc])


def _binarisation(image_loc, dest_image_loc):
    subprocess.check_call(_command_tesseract[:3] + [image_loc, image_loc] + _command_tesseract[3:])


def binarisation(image_loc, dest_image_loc):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    _convert(image_loc, dest_image_loc)
    # _convert(dest_image_loc, dest_image_loc)
    _binarisation(dest_image_loc, dest_image_loc)


# noinspection PyUnusedLocal
def blur(image_loc, dest_image_loc):
    pass


def main(image_loc, dest_image_loc=None):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    print(image_loc)
    if 1:  # dest_image_loc is None:
        filename = os.path.basename(image_loc)
        dest_image_loc = os.path.join("/home/sampathm/Downloads/InvoiceData/FY1820001_PNG/", filename) + '.jpg'

    if os.path.isfile(dest_image_loc):
        print('Binarisation found existing file {}'.format(dest_image_loc))
        return
    try:
        binarisation(image_loc, dest_image_loc)
        print('Binarisation generated file {}'.format(dest_image_loc))
    except Exception as e:
        print(e)
        print('Binarisation - Failed - Generated file {}'.format(dest_image_loc))


class PdfToOCR(ModuleInterface):

    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = "/home/sampathm/Downloads/InvoiceData/hyperion/"  # config.IMAGE_ROOT_DIR
        self.destination_folder = "/home/sampathm/Downloads/InvoiceData/hyperion/"  # config.BINARIZE_ROOT_DIR
        # Transformation function for converting source_image to destination_image
        self.operator_func = main

    def bulk_run(self):
        if (not self.source_folder) or (not self.operator_func):
            raise RuntimeError('self.source_folder or self.operator_func is not defined !!')
        import glob
        all_images = glob.glob(self.source_folder + '/*pdf')
        print(all_images)
        for each in all_images:
            print(each, '--- ' * 5)
            main(each)


if __name__ == '__main__':
    t = PdfToOCR()
    t.plugin_inputs()
    print('--' * 55)
    t.bulk_run()
