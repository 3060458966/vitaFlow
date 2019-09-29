# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np

class ImagePreproccessorInterface:
    def __init__(self):
        pass

    def resize(self, image, height, width):
        raise NotImplementedError

    def binarization(self, image):
        raise NotImplementedError

    def deskew(self, image):
        raise NotImplementedError

    def rotate(self, image, angle):
        raise NotImplementedError

    def stiching(self, images):
        raise NotImplementedError

    def sharpening(self, image):
        raise NotImplementedError

class OpenCV(ImagePreproccessorInterface):
    def __init__(self):
        ImagePreproccessorInterface.__init__(self)

    def resize(self, image, height, width):
        return cv2.resize(image, (width, height))

    def rotate(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def binarization(self, image, maxval=255, blockSize=31, C=2):
        """
        :param image:
        :param maxval:
        :param blockSize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        :param C: C Constant subtracted from the mean or weighted mean (see the details below). Normally, it
    .   is positive but may be zero or negative as well.
        :return:
        """

        # ret,thresh_img = cv2.threshold(image,thresh,maxval,cv2.THRESH_BINARY)
        thresh_img = cv2.adaptiveThreshold(image, maxval,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, blockSize, C)
        return thresh_img

    def blur(self, image, kernel_size=(3,3)):
        return cv2.GaussianBlur(image, kernel_size, 0)

