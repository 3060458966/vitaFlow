import multiprocessing
import sys
import os
sys.path.append(os.getcwd())
from glob import glob
from functools import partial

from vitaflow.utils.print_helper import print_error

import fire
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_files_with_ext(search_folder, exts=['.JPG', '.jpg', '.png']):
    all_files = glob(search_folder + '**/**', recursive=True)
    bag = []
    if exts:
        for _ext in exts:
            bag += [file for file in all_files if file.endswith(_ext)]
    else:
        bag = all_files
    return bag


def get_all_input_files(source_dir, input_files_types=['.JPG', '.jpg', '.png']):
    """Get the list of images files from the source directory"""
    return find_files_with_ext(source_dir, input_files_types)


def crop_and_save(cords, image, file_path):
    (x1, x2, y1, y2) = cords
    cropped_image = image[y1:y2, x1:x2]
    dest_file = file_path + ".png"
    try:
        plt.imsave(dest_file, cropped_image, cmap='Greys_r')
    except Exception as e:
        print_error('Missed to write file  {} : {e}'.format(dest_file, e))


def crop_to_box(image_file_path, destination_dir, text_file_ext):
    try:
        gt_text_file_path = image_file_path.replace(".jpg", ".txt")
        source_image_path = image_file_path
        file_name = source_image_path.split("/")[-1].split(".")[0]
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        # Open the text file and get all the coordinates
        with open(gt_text_file_path) as gt_txt_file_pointer:
            count = 0
            coords_data = []
            for line in gt_txt_file_pointer:
                coords_data.append(line.strip().split(","))
            for gt_txt_line in coords_data:
                try:
                    # print(gt_txt_line)
                    jpgfile = plt.imread(source_image_path)
                    # naming convention for the file
                    out_file_path = destination_dir +"/" + file_name + "_" + str(count)

                    # x1, y1, _, _, x2, y2, _, _, text = gt_txt_line
                    x1 = gt_txt_line[0]
                    y1 = gt_txt_line[1]
                    x2 = gt_txt_line[4]
                    y2 = gt_txt_line[5]
                    text = "".join(gt_txt_line[8:])
                    # call fun with cords and images named convention for the cropped image
                    crop_and_save((int(x1), int(x2), int(y1), int(y2)), jpgfile, out_file_path)  # (int(x1)-11, int(x2)+11, int(y1)-4, int(y2)+4
                    count = count + 1
                    with open(out_file_path+text_file_ext, "w") as fd:
                        fd.write(text)
                except FileNotFoundError as fnf_error:
                    print("error", fnf_error)
    except Exception as e:
        print(e)
        print(image_file_path)


def prepare_calamari_dataset_from_icdar(in_path, out_path="cropped",
                                        text_file_ext=".gt.txt"):
    """
    Perpares the dataset for calamari training from ICDAR based train dataset
    :param in_path: Directory with images and corresponding text files
    :param out_path: Directory where
    :param text_file_ext: Text file extension [.txt, .gt.txt]
    :return:
    """
    in_files = get_all_input_files(source_dir=in_path)
    crop_to_box_partial = partial(crop_to_box, destination_dir=out_path, text_file_ext=text_file_ext)

    with multiprocessing.Pool() as p:
        r = list(tqdm(p.imap(crop_to_box_partial, in_files), total=len(in_files)))
        p.close()
        p.join()


if __name__ == '__main__':
    fire.Fire(prepare_calamari_dataset_from_icdar)