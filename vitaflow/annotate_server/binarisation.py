import subprocess


def _convert(image_loc, dest_image_loc):
    subprocess.check_call(['/usr/local/bin/convert',
                           '-auto-level',
                           '-sharpen',
                           '0x4.0',
                           '-contrast',
                           image_loc, dest_image_loc])


def _binarisation(image_loc, dest_image_loc):
    subprocess.check_call(['./textcleaner',
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
                           '10',
                           dest_image_loc, dest_image_loc])


def binarisation(image_loc, dest_image_loc):
    # print('binarisation src {} dest {} '.format(image_loc, dest_image_loc))
    # TODO: experiment & optimize below
    subprocess.check_call(['/usr/local/bin/convert',
                           '-auto-level',
                           '-sharpen',
                           '0x4.0',
                           '-contrast',
                           image_loc, dest_image_loc])
    subprocess.check_call(['./textcleaner',
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
                           '10',
                           dest_image_loc, dest_image_loc])
    subprocess.check_call(['/usr/local/bin/convert',
                           '-auto-level',
                           '-sharpen',
                           '0x4.0',
                           '-contrast',
                           dest_image_loc, dest_image_loc])
    print('binersaisation Generated file {}'.format(dest_image_loc))


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
        print('bineraisation Found existing file {}'.format(dest_image_loc))
        return
    try:
        binarisation(image_loc, dest_image_loc)
        print('bineraisation Generated file {}'.format(dest_image_loc))
    except:
        print('bineraisation - Failed - Generated file {}'.format(dest_image_loc))


from .bin.plugin import pluginApplication
from common import verify_image_ext, verify_input_file


class imageBineraisePlugin(pluginApplication):
    def inputs(self, image_loc, dest_image_loc=None):
        self._inputs = (image_loc, dest_image_loc)

        if not all([
            verify_input_file(image_loc),
            verify_image_ext(image_loc)
        ]):
            raise ValueError('inputs are ')
        if dest_image_loc is None:
            filename = os.path.basename(image_loc)
            dest_image_loc = os.path.join(config.ROOT_DIR, config.BINARIZE_ROOT_DIR, filename)

        self._inputs_validated = True

    def run(self):
        self.validate_inputs()
        binarisation(*self._inputs)


if __name__ == '__main__':
    from glob import glob
    import os
    import config

    raw_images = glob(os.path.join(config.IMAGE_ROOT_DIR + '/*jpg'))
    raw_images = sorted(raw_images)
    from multiprocessing import Pool

    # for each in raw_images:
    #     filename = os.path.basename(each)
    #     new_file_name = os.path.join(config.ROOT_DIR, config.BINARIZE_ROOT_DIR, filename)
    #     src_file_name = os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR, filename)
    #     binarisation(src_file_name, new_file_name)
    all_src_images = [
        os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR, os.path.basename(each)) for each in raw_images
    ]
    with Pool(5) as p:
        print(p.map(main, all_src_images))
