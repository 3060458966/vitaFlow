# coding=utf-8
from __future__ import unicode_literals

"""
Convert `Image` to `Text Line Images` and then convert them to `Text` file.  

to run
    `PYTHONIOENCODING=utf-8 python3`

"""

import cv2
from bin.plugin import pluginApplication
from common import verify_input_file, verify_image_ext
from image_processing import get_line_segments
from bin.utils import trim_file_ext
import matplotlib.pyplot as plt


# def image_to_text_lines_images(image):
#     line_segments = get_line_segments(image)
#     collected_text_data = []
#     i = 1
#     for start, end in line_segments:
#         if abs(start - end) < 10:
#             continue
#         text_image = image[start - 2: end + 2, :]
#
#         text_data = image_ocr(text_image)
#         if text_data:
#             # show_img(image[start - 1: end + 1, :])
#             # print(text_data)
#             collected_text_data.append(text_data)
#     return collected_text_data


def main(image_filename):
    """
    convert image -> text file

    :param image_filename: filename with path
    :return:
    """
    _text_file_name = (os.path.basename(image_filename)).rsplit('.')[0] + '.txt'
    text_file_name = os.path.join(config.TEXT_DIR, _text_file_name)
    # if os.path.isfile(text_file_name):
    #     print('Found the text file for {}'.format(image_filename))
    #     return
    # try:
    #     image = cv2.imread(image_filename, 0)
    #     collected_text_data = image_to_text_lines_images(image)
    # except:
    #     collected_text_data = []
    #     print('Failed - Image2Text {} {}'.format(image_filename, text_file_name))
    # with open(text_file_name, 'w', encoding='utf-8') as fp:
    #     fp.write(u'\n'.join(collected_text_data))
    #     # import pdb
    #     # pdb.set_trace()
    #     # print(u'\n'.join(collected_text_data))
    #     print('Image2Text {} {}'.format(image_filename, text_file_name))
    source_image = image_filename
    image = cv2.imread(source_image, 0)
    line_segments = get_line_segments(image)
    i = 1
    image_dir = os.path.join(config.TEXT_IMAGES, trim_file_ext(os.path.basename(source_image)))
    os.mkdir(image_dir)
    for start, end in line_segments:
        if abs(start - end) < 10:
            continue
        text_image = image[start - 2: end + 2, :]
        plt.imsave(os.path.join(image_dir, str(i) + '.png'), text_image)
        i += 1


class imageToTextImagesPlugin(pluginApplication):
    '''converts Input Image to '''

    def inputs(self, source_image, desct_file=None):
        validation_fns = [
            verify_input_file,
            verify_image_ext
        ]
        if not all([f(source_image) for f in validation_fns]):
            raise ValueError('Source Image {} Failed validation fns')
        if desct_file is None:
            _desct_file = (os.path.basename(source_image)).rsplit('.')[0] + '.txt'
            desct_file = os.path.join(config.TEXT_DIR, _desct_file)

        self._inputs = (source_image, desct_file)
        self._input_validated = True

    def run(self):
        self.validate_inputs()
        (source_image, desct_file) = self._inputs
        # try:
        #     image = cv2.imread(source_image, 0)
        #     collected_text_data = image_to_text_lines_images(image)
        # except:
        #     collected_text_data = []
        #     print('Failed - Image2Text {}'.format(source_image))
        # with open(desct_file, 'w', encoding='utf-8') as fp:
        #     fp.write(u'\n'.join(collected_text_data))
        #     print('Image2Text {} {}'.format(source_image, desct_file))
        image = cv2.imread(source_image, 0)
        line_segments = get_line_segments(image)
        i = 1
        image_dir = os.path.join(config.TEXT_IMAGES, trim_file_ext(os.path.basename(source_image)))
        if os.path.isdir():
            print('Dir already exists {}'.format(image_dir))
            return
        os.mkdir(image_dir)
        print('created {}'.format(image_dir))
        for start, end in line_segments:
            if abs(start - end) < 10:
                continue
            text_image = image[start - 2: end + 2, :]
            text_image_fname = os.path.join(image_dir, str(i) + '.png')
            plt.imsave(text_image_fname, text_image)
            i += 1


# def async_main():
#     import asyncio
#     import random
#     dict = {'router1': {'id': 1, 'name': 'rtr1_core'},
#             'router2': {'id': 2, 'name': 'rt2_core'},
#             'router3': {'id': 3, 'name': 'rtr3_access'}}
#
#     async def process_rtr(id, name):
#         """Do your execution here."""
#         s_time = await asyncio.sleep(random.randint(0, 5))
#         print(f"Processing {id}, {name}")
#
#     loop = asyncio.get_event_loop()
#     tasks = [asyncio.ensure_future(process_rtr(**router_details))
#              for router, router_details
#              in dict.items()]
#     loop.run_until_complete(asyncio.wait(tasks))
#     loop.close()


if __name__ == '__main__':
    from glob import glob
    import os
    import config
    from multiprocessing import Pool

    raw_images = glob(os.path.join(config.BINARIZE_ROOT_DIR + '/*jpg'))
    raw_images = sorted(raw_images)
    multi_threading = False
    if not multi_threading:
        for each in raw_images:
            print(each)
            main(each)
    else:
        with Pool(5) as p:
            print(p.map(main, raw_images))
    # im2tl = imageToTextImagesPlugin()
    # im2tl.quick_run()
