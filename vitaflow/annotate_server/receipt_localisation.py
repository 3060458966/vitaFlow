"""
Receipt Localisation using East

Added East data processing code for receipt localisation

Using images & east generated text files in East folder,
image files are processed and save to Images folder.

"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from bin.plugin import PluginAppModel
from bin.utils import trim_file_ext


def get_slope(x2, x3, y2, y3):
    if x3 - x2 == 0:
        return 'infinity'
    return ((y3 - y2) * 1.0) / (x3 - x2)


def read_data(text_filename):
    raw_data = []
    bag = []
    data = open(text_filename).read().splitlines()
    for line in data:
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(round, map(float, line.split(','))))
        raw_data.append((x1, y1, x2, y2, x3, y3, x4, y4))
        _x = [x1, x2, x3, x4]
        _y = [y1, y2, y3, y4]
        x_max = max(_x)
        x_min = min(_x)
        y_max = max(_y)
        y_min = min(_y)
        coords = [y_min, y_max, x_min, x_max]
        bag.append(coords)
    return raw_data, bag


def get_box_slaves(x1, y1, x2, y2, x3, y3, x4, y4):
    slopes = (get_slope(x1, x2, y1, y2),
              get_slope(x2, x3, y2, y3),
              get_slope(x3, x4, y3, y4),
              get_slope(x4, x1, y4, y1))
    return slopes


def get_all_slopes(raw_data):
    for x1, y1, x2, y2, x3, y3, x4, y4 in raw_data:
        slopes = get_box_slaves(x1, y1, x2, y2, x3, y3, x4, y4)
        print(slopes)


def get_plot_points(raw_data):
    plot_points = []
    for x1, y1, x2, y2, x3, y3, x4, y4 in raw_data:
        plot_points += [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
    return plot_points


def length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_area(x1, y1, x2, y2, x3, y3, x4, y4):
    lenghts = [length(x1, y1, x2, y2),
               length(x2, y2, x3, y3),
               length(x3, y3, x4, y4),
               length(x4, y4, x1, y1)
               ]
    side_a = (lenghts[0] + lenghts[2]) / 2
    side_b = (lenghts[1] + lenghts[3]) / 2
    return side_a, side_b, round(side_a * side_b, 2)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def trim_outliers1(raw_data):
    new_plot_points = []
    stats = []
    plot_points = []
    mins_box_lengths = []

    for each in raw_data:
        x1, y1, x2, y2, x3, y3, x4, y4 = each
        _points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
        plot_points += _points
        side_a, side_b, tot_area = get_area(x1, y1, x2, y2, x3, y3, x4, y4)
        stats.append((side_a, side_b, tot_area))
        mins_box_lengths.append(min(side_a, side_b))

    _acceptable_min_length = np.quantile(sorted(mins_box_lengths), 0.50)
    # TODO - remove - too big or too small
    _acceptable_min_length = _acceptable_min_length * 2.5

    # find median - text length
    for i in range(len(stats)):
        side_a, side_b, tot_area = stats[i]
        if min((side_a, side_b)) < _acceptable_min_length:
            x1, y1, x2, y2, x3, y3, x4, y4 = raw_data[i]
            new_plot_points += [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
    return new_plot_points


def rotate_image_with_east(img_filename, save_file=None):
    image = plt.imread(img_filename)
    text_filename = os.path.join(
        os.path.dirname(img_filename),
        trim_file_ext(os.path.basename(img_filename)) + '.txt')
    raw_data, bag = read_data(text_filename)
    new_plot_points = trim_outliers1(raw_data)
    cnt = np.array(new_plot_points)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    warped = four_point_transform(image, box)
    if save_file:
        plt.imsave(save_file, warped)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(warped)


class receiptLocalisationPlugin(PluginAppModel):

    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.EAST_DIR
        self.destination_folder = config.IMAGE_ROOT_DIR
        # Transformation function for converting source_image to destination_image
        self.operator_func = rotate_image_with_east


#
#
# if __name__ == '__main__':
#     import traceback
#
#     raw_images = glob(os.path.join(config.EAST_DIR + '/*jpg'))
#     raw_images = sorted(raw_images)
#     for each in raw_images:
#         #     break
#         if ('_act' in each) or ('_predict' in each):
#             continue
#         error_log = []
#         filename = os.path.basename(each)
#         new_file_name = os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR, filename)
#         text_file = os.path.join(
#             os.path.dirname(each),
#             trim_file_ext(os.path.basename(each)) + '.txt')
#
#         message = '--------  ' * 5
#         message += 'Not able to find text file for {}'.format(each)
#         if os.path.isfile(text_file):
#             try:
#                 if not os.path.isfile(new_file_name):
#                     rotate_image_with_east(each, new_file_name)
#                 else:
#                     print('File is already present {}'.format(each))
#             except Exception as err:
#                 message = '--------  ' * 5
#                 message += 'Error while running for {}'.format(each)
#                 error_log.append(message)
#                 traceback.print_tb(err.__traceback__)
#                 error_log.append(str(traceback.format_exc()))
#         else:
#             print(message)
#             error_log.append(message)
#     with open('receipt_localisation.log', 'a+') as fp:
#         fp.writelines(error_log)


if __name__ == '__main__':
    t = receiptLocalisationPlugin()
    t.plugin_inputs()
    print('--' * 55)
    t.bulk_run()
