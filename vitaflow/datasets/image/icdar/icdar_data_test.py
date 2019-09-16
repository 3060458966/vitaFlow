import os
import sys
# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))
import unittest
from vitaflow.datasets.image.icdar.icdar_data import CDARDataset


class CDARDatasetTest(unittest.TestCase):

    # Returns True or False.
    def test(self):
        dataset = CDARDataset(data_in_dir="/opt/vlab/icdar-2019-data",
                              data_out_dir="/opt/tmp/vitaflow/east_v2_keras/data/",
                              is_preprocess=False,
                              max_image_large_side=1280,
                              max_text_size=800,
                              min_text_size=5,
                              min_crop_side_ratio=0.1,
                              geometry="RBOX",
                              number_images_per_tfrecords=8,
                              num_cores=4,
                              batch_size=4,
                              prefetch_size=16)

        assert (dataset.train_samples_count == 547)

        gen = dataset.get_train_dataset_gen()
        i = 0
        for features_n_label in next(gen):
            print(features_n_label[0][0].shape)
            i += 1
            if i == 2:
                break

if __name__ == '__main__':
    unittest.main()