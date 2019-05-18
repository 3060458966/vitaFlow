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


def crop_and_save(cords, image, dest, fname):
    (x1, x2, y1, y2) = cords

    cropped_image = image[y1:y2, x1:x2]
    dest_file = os.path.join(dest, fname)
    plt.imsave(dest_file, cropped_image, cmap='Greys_r')  # '
    print('Saved file to {}'.format(dest_file))


def main(source_image_loc, dest_image_loc):

    source_loc_dir = os.path.dirname(source_image_loc)

    dest_loc_dir = os.path.dirname(dest_image_loc)


    gt_text_base = os.path.basename(source_image_loc).split(".")[0]
    gt_text_name = gt_text_base + ".txt"

    cropped_dir=os.path.join(dest_loc_dir,gt_text_base)
    if not os.path.isdir(cropped_dir):
        os.mkdir(os.path.join(cropped_dir))
    #GT this config injection is not a very good approach
    gt_text_file_loc = os.path.join(config.ROOT_DIR,config.EAST_DIR,gt_text_name)
    if not os.path.isfile(gt_text_file_loc):
        print("Skipping the run as {} has not east predictions".format(gt_text_file_loc))
    else:
        gt_image_base = os.path.basename(source_image_loc).split(".")[0]

        # Open the text file and get all the coordinates
        with open(gt_text_file_loc) as gt_txt_file:
            count = 0
            # for every cords for the image
            for gt_txt_line in gt_txt_file:
                gt_txt_line = gt_txt_line.strip()
                x1, y1, _, _, x2, y2, _, _ = gt_txt_line.split(",")
                try:
                    jpgfile = plt.imread(source_image_loc)

                    # naming convention for the file
                    image_name = gt_image_base + "_" + str(count)
                    # call fun with cords and imagesame named convention for the cropped image
                    crop_and_save((int(x1), int(x2), int(y1), int(y2)), jpgfile, cropped_dir,image_name)  # (int(x1)-11, int(x2)+11, int(y1)-4, int(y2)+4
                    count = count + 1
                except FileNotFoundError as fnf_error:
                    print("error", fnf_error)


class cropToBoxPlugin(PluginAppModel):

    def plugin_inputs(self):
        # Custom location according to need
        self.source_folder = config.BINARIZE_ROOT_DIR
        self.destination_folder = config.TEXT_IMAGES
        # Transformation function for converting source_image to destination_image
        self.operator_func = main



if __name__ == '__main__':
    t = cropToBoxPlugin()
    t.plugin_inputs()
    print('--' * 55)
    t.bulk_run()
