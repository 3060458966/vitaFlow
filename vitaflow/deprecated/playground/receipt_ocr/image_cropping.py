import matplotlib.pyplot as plt
import os


class ImageCropping:
    """
    To split Image to single line text Images.

    Receipt(.png) to (Merchant, Date, Line Item, Tax, Total)(.png)
    """

    def __init__(self, image_loc, dest=None):
        # verify if image file exists
        if not os.path.isfile:
            raise 'FileNotFound {}'.format(image_loc)
        self.image = plt.imread(image_loc)
        # image loc name
        self.source, self.fname = os.path.split(image_loc)
        # dest - location preparation
        self.dest = dest
        self._newfolder = self.fname.rsplit('.')[0]
        self._ext = self.fname.rsplit('.')[1]
        if self.dest is None:
            self.dest = os.path.join(self.source, self._newfolder)
        else:
            if not os.path.isdir(self.dest):
                os.mkdir(self.dest)
            self.dest = os.path.join(self.dest, self._newfolder)
        if not os.path.isdir(self.dest):
            os.mkdir(self.dest)

    def crop_and_save(self, cords, fname):
        (x1, x2, y1, y2) = cords
        cropped_image = self.image[x1:x2, y1:y2]
        if not fname.endswith(self._ext):
            print('Adding ' + self._ext)
            fname = fname + '.' + self._ext
        dest_file = os.path.join(self.dest, fname)
        plt.imsave(dest_file, cropped_image)
        print('Saved file to {}'.format(dest_file))

    def multiple_crop_and_save(self, cords_list, labels_list):
        i = 0
        for cords, label in zip(cords_list, labels_list):
            self.crop_and_save(cords, '{}_{}.{}'.format(label, i, self._ext))
            i += 1

    def show_image(self, cords=None):
        if cords:
            (x1, x2, y1, y2) = cords
            plt.imshow(self.image[x1:x2, y1:y2])
        else:
            plt.imshow(self.image)
