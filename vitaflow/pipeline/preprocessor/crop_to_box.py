"""
Receipt Localisation using East

Added East data processing code for receipt localisation

Using images & east_airflow_demo generated text files in East folder,
image files are processed and save to Images folder.

"""

import os

import matplotlib.pyplot as plt

from vitaflow.pipeline.interfaces.plugin import ImagePluginInterface
from vitaflow import demo_config


def crop_and_save(cords, image, dest, fname):
    (x1, x2, y1, y2) = cords
    cropped_image = image[y1:y2, x1:x2]
    dest_file = os.path.join(dest, fname)
    dest_file = dest_file + ".jpg"
    try:
        plt.imsave(dest_file, cropped_image, cmap='Greys_r')  # '
        print('Saved file to {}'.format(dest_file))
    except:
        print('>>>>>>>>>>>>>> Missed file to {}'.format(dest_file))


def sorting_east_cords_data(gt_txt_file_pointer):
    """Sorts the data according to the locations in the """
    new_data = []
    for line in gt_txt_file_pointer:
        new_data.append(list(map(int, line.strip().split(","))))

    def cmp_fns_x(cords):
        """sorting with respect to x-axis"""
        x1, y1, _, _, x2, y2, _, _ = cords
        return x1

    def cmp_fns_y(cords):
        """sorting with respect to y-axis"""
        x1, y1, _, _, x2, y2, _, _ = cords
        return y1

    new_data = sorted(new_data, key=cmp_fns_x)
    new_data = sorted(new_data, key=cmp_fns_y)
    return new_data


def crop_to_box(gt_text_file_loc, source_image_loc, cropped_dir):
    # Open the text file and get all the coordinates
    with open(gt_text_file_loc) as gt_txt_file_pointer:
        count = 0
        sorted_gt_txt_data = sorting_east_cords_data(gt_txt_file_pointer)
        for gt_txt_line in sorted_gt_txt_data:
            try:
                jpgfile = plt.imread(source_image_loc)
                # naming convention for the file
                image_name = str(count)
                x1, y1, _, _, x2, y2, _, _ = gt_txt_line
                # call fun with cords and images named convention for the cropped image
                crop_and_save((int(x1), int(x2), int(y1), int(y2)), jpgfile, cropped_dir,
                              image_name)  # (int(x1)-11, int(x2)+11, int(y1)-4, int(y2)+4
                count = count + 1
            except FileNotFoundError as fnf_error:
                print("error", fnf_error)


class EastCropperImagePlugin(ImagePluginInterface):
    def __init__(self, east_out_dir):
        ImagePluginInterface.__init__(self)
        self._east_out_dir = east_out_dir

    def _handle_data(self, in_file_data):
        """Each plugin module should implement this to handle image array data"""
        raise NotImplementedError

    def _handle_file(self, in_file_path, out_file_path):
        source_loc_dir = os.path.dirname(in_file_path)
        dest_loc_dir = os.path.dirname(out_file_path.split(".")[0] + "/")
        gt_text_base = os.path.basename(in_file_path).split(".")[0]
        gt_text_name = gt_text_base + ".txt"
        if not os.path.isdir(dest_loc_dir):
            os.mkdir(os.path.join(dest_loc_dir))

        gt_text_file_loc = os.path.join(self._east_out_dir, gt_text_name)

        if not os.path.isfile(gt_text_file_loc):
            print("Skipping the run as {} has not east_airflow_demo predictions".format(gt_text_file_loc))
        else:
            crop_to_box(gt_text_file_loc, in_file_path, dest_loc_dir)


if __name__ == '__main__':
    t = EastCropperImagePlugin(east_out_dir=demo_config.EAST_OUT_DIR)
    print('--' * 55)
    t.process_files(source_dir=demo_config.BINARIZE_ROOT_DIR, destination_dir=demo_config.CROPPER_ROOT_DIR)
