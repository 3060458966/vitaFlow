import os
import tempfile
from glob import glob

import cv2
import pytesseract
from matplotlib import pyplot as plt

import config
from bin.plugin import textExtraction


def tesseract_ocr(image_filename):
    "A single text line image is provided for extracted text"
    image = cv2.imread(image_filename, 0)
    return pytesseract.image_to_string(image, config=config.TESSERACT_CONFIG)


def calamari_ocr(image):
    fd, filename = tempfile.mkstemp(suffix='.png')
    try:
        plt.imsave()
        os.write(fd, someStuff)
        os.close(fd)
        # ...run the subprocess and wait for it to complete...
    finally:
        os.remove(filename)


image_ocr = tesseract_ocr


class ocrTesseract(textExtraction):

    def inputs(self):
        self._inputs = glob(os.path.join(config.TEXT_IMAGES, '*'))[:10]
        print('collected files {}'.format(len(self._inputs)))

    def run(self):
        for image_dir in self._inputs:
            _image_files = glob(os.path.join(image_dir, '*'))
            _image_files = sorted(_image_files)

            _text_file_name = image_dir + '.txt'
            text_file_name = os.path.join(config.TEXT_DIR, os.path.basename(_text_file_name))
            if os.path.isfile(text_file_name):
                print('Found the text file for {}'.format(_text_file_name))
            else:
                collected_text_data = []
                for _line_image in _image_files:
                    collected_text_data.append(tesseract_ocr(_line_image))
                with open(text_file_name, 'w', encoding='utf-8') as fp:
                    fp.write(u'\n'.join(collected_text_data))
                print('File: {}'.format(text_file_name))


if __name__ == '__main__':
    t = ocrTesseract()
    t.quick_run()
