import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image


###########################################################
# Step 1.
def seggregate_images():
    processed_data_out_dirs = ["./data-valid", "./data-invalid"]
    # create the data-sorted folder if not exists
    for folder in processed_data_out_dirs:
        if os.path.exists(folder):
            # print_info("Deleting data folder: {}".format(folder))
            shutil.rmtree(folder)
            # print_info("Recreating data folder: {}".format(folder))
            os.makedirs(folder)
        else:
            # print_info("Creating data folder: {}".format(folder))
            os.makedirs(folder)
    # loop through the files from folder and classify them into valid/invalid
    source_folder = "data-200"
    for image_file in os.listdir(source_folder):
        print(image_file)
        im = Image.open(os.path.join(source_folder, image_file))
        #
        print(im.size)

        if im.size[0] < 320 or im.size[1] < 320:
            jpgfile = os.path.join(source_folder, image_file)
            dst_dir = "data-invalid"
            # dst_dir = os.path.join(dst_dir,jpgfile)
            shutil.copy(jpgfile, dst_dir)
        else:
            jpgfile = os.path.join(source_folder, image_file)
            dst_dir = "data-valid"
            # dst_dir = os.path.join(dst_dir, jpgfile)
            shutil.copy(jpgfile, dst_dir)


# seggregate_images()
# data-200 -> data-valid
#          -> data-invalid
###########################################################
# Step 2 Perform Character Detection
# data-valid -> original_image/prediction/activation/txt
#
###########################################################
# Step 3 Detect the text characters and select the best boxes
# for drawing the boundary

# def draw_bounding_boxes():
import glob

folder_dir = "./data-valid"
original_images = []

for file_instance in os.listdir(folder_dir):
    # print("\n",file_instance)
    if len(file_instance.split(".")) > 2:
        continue
        # print("It is a non image",file_instance)
    else:
        if ".txt" in file_instance:
            # print("It is a text file",file_instance)
            continue
        else:
            # print("Normal Image",file_instance)
            original_images.append(file_instance)

##### Done with getting the file names of the original images
import numpy as np
import sys
import cv2
from pyimagesearch.transform import four_point_transform

# Load an color image in grayscale

for img_instance in original_images:
    # img_instance ="348s (145).jpg"
    bounding_file = os.path.join("data-valid", os.path.splitext(img_instance)[0] + ".txt")

    image = plt.imread(os.path.join("data-valid", img_instance))

    if not os.path.exists(bounding_file):
        print("No file exists", bounding_file)
        sys.exit(1)

    # Read the txt file containing the co-ordinates
    with open(bounding_file) as f:
        data = f.read().splitlines()

        bag = []
        for line in data:
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(round, map(float, line.split(','))))
            _x = [x1, x2, x3, x4]
            _y = [y1, y2, y3, y4]
            x_max = max(_x)
            x_min = min(_x)
            y_max = max(_y)
            y_min = min(_y)
            coords = [y_min, y_max, x_min, x_max]
            bag.append(coords)

        bag = np.array(bag)
        # print(bag)
        min_cords = np.amin(bag, axis=0)
        # print(min_cords)
        max_cords = np.amax(bag, axis=0)

        # print(max_cords)
        boundin_y_min = min_cords[0] if min_cords[0] > 0 else 0
        boundin_y_max = max_cords[1] if max_cords[1] > 0 else 0
        boundin_x_min = min_cords[2] if min_cords[2] > 0 else 0
        boundin_x_max = max_cords[3] if max_cords[3] > 0 else 0

        print(boundin_y_min, boundin_y_max, boundin_x_min, boundin_x_max)
        print(image.shape)
        print(img_instance)

        receipt_detected = image[boundin_y_min:boundin_y_max, boundin_x_min:boundin_x_max]
        # plt.imshow(image[boundin_y_min:boundin_y_max, boundin_x_min:boundin_x_max])
        # plt.show()
        plt.imsave(os.path.join('./data-detected', img_instance), receipt_detected)

        # get the

print(len(original_images))
