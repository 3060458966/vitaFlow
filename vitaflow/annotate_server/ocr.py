import pytesseract

import config


def image_ocr(image):
    "A single text line image is provided for extracted text"
    return pytesseract.image_to_string(image, config=config.TESSERACT_CONFIG)
