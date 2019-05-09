# coding=utf-8
from __future__ import unicode_literals
"""
Convert `Image` to `Text Line Images` and then convert them to `Text` file.  

to run
    `PYTHONIOENCODING=utf-8 python3`

"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from bin.plugin import PluginAppModel
from bin.utils import trim_file_ext
import config


def show_img(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(data, 'gray')


def get_threshold_image(image):
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_threshold_image2(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_threshold_image3(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def get_line_segments(image):
    # threshold
    image = _avail_thershold_fns[_selected_threshold_fns](image)
    plt_data = image.min(axis=1)
    # plt.figure(figsize=(15, 2))
    # plt.plot(range(len(plt_data)), plt_data, '*')
    plt_data_index = np.arange(len(plt_data))
    data = plt_data_index[plt_data == 0]
    i = 0
    start = i
    memory = data[i]

    line_segments = []

    while i < len(data) - 1:
        i += 1
        if data[i] == memory + 1:
            memory += 1
        else:
            line_segments.append(
                (data[start], data[i])
            )
            # print(data[start], data[i])
            start = i
            memory = data[i]
    line_segments.append((data[start], data[i]))
    return line_segments


def main(image_filename, image_dir=None):
    """
    convert image -> text file

    :param image_filename: filename with path
    :return:
    """
    source_image = image_filename
    image = cv2.imread(source_image, 0)
    line_segments = get_line_segments(image)
    line_image_offset = 0
    i = 1
    if not image_dir:
        image_dir = os.path.join(config.TEXT_IMAGES, trim_file_ext(os.path.basename(source_image)))
    if os.path.isdir(image_dir):
        print('Skipping the run as {} already exists'.format(image_dir))
        return
    else:
        os.mkdir(image_dir)
    for start, end in line_segments:
        if abs(start - end) < 10:
            continue
        text_image = image[start - line_image_offset: end + line_image_offset, :]
        plt.imsave(os.path.join(image_dir, str(i) + '.png'), text_image)
        i += 1


#
#
# class imageToTextImagesPlugin(pluginApplication):
#     '''converts Input Image to '''
#     def inputs(self, source_image, desct_file=None):
#         validation_fns = [
#             verify_isfile,
#             verify_isimagefile
#         ]
#         if not all([f(source_image) for f in validation_fns]):
#             raise ValueError('Source Image {} Failed validation fns')
#         if desct_file is None:
#             _desct_file = (os.path.basename(source_image)).rsplit('.')[0] + '.txt'
#             desct_file = os.path.join(config.TEXT_DIR, _desct_file)
#
#         self._inputs = (source_image, desct_file)
#         self._input_validated = True
#
#     def run(self):
#         self._validate_inputs()
#         (source_image, desct_file) = self._inputs
#         # try:
#         #     image = cv2.imread(source_image, 0)
#         #     collected_text_data = image_to_text_lines_images(image)
#         # except:
#         #     collected_text_data = []
#         #     print('Failed - Image2Text {}'.format(source_image))
#         # with open(desct_file, 'w', encoding='utf-8') as fp:
#         #     fp.write(u'\n'.join(collected_text_data))
#         #     print('Image2Text {} {}'.format(source_image, desct_file))
#         image = cv2.imread(source_image, 0)
#         line_segments = get_line_segments(image)
#         i = 1
#         image_dir = os.path.join(config.TEXT_IMAGES, trim_file_ext(os.path.basename(source_image)))
#         if os.path.isdir():
#             print('Dir already exists {}'.format(image_dir))
#             return
#         os.mkdir(image_dir)
#         print('created {}'.format(image_dir))
#         for start, end in line_segments:
#             if abs(start - end) < 10:
#                 continue
#             text_image = image[start - 2: end + 2, :]
#             text_image_fname = os.path.join(image_dir, str(i) + '.png')
#             plt.imsave(text_image_fname, text_image)
#             i += 1


# if __name__ == '__main__':
#     from glob import glob
#     import os
#     import config
#     from multiprocessing import Pool
#
#     raw_images = glob(os.path.join(config.BINARIZE_ROOT_DIR + '/*jpg'))
#     raw_images = sorted(raw_images)
#     multi_threading = False
#     if not multi_threading:
#         for each in raw_images:
#             print(each)
#             main(each)
#     else:
#         with Pool(5) as p:
#             print(p.map(main, raw_images))
#     # im2tl = imageToTextImagesPlugin()
#     # im2tl.quick_run()


class imageToTextLinesImages(PluginAppModel):

    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.BINARIZE_ROOT_DIR
        self.destination_folder = config.TEXT_IMAGES
        # Transformation function for converting source_image to destination_image
        self.operator_func = main

    def run(self):
        """Execute of Code logic"""
        source, dest = self._inputs
        dest = trim_file_ext(self._inputs[1])
        print('New Dest {}'.format(dest))
        self._inputs = (source, dest)
        super().run()


# Global Settings
_avail_thershold_fns = {
    'binary': get_threshold_image,
    'adaptgaussian': get_threshold_image2,
    'gaussian_otsu': get_threshold_image3
}
_selected_threshold_fns = 'binary'


if __name__ == '__main__':
    t = imageToTextLinesImages()
    t.plugin_inputs()
    print('--' * 55)
    t.bulk_run()
