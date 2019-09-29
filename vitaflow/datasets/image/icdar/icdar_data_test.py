import os
import sys
# Appending vitaFlow main Path
from tensorflow_core._api.v2 import errors

sys.path.append(os.path.abspath('.'))
import unittest
from vitaflow.datasets.image.icdar.icdar_data import CDARDataset


def _get_next_batch(generator):
  """Retrieves the next batch of input data."""
  try:
    generator_output = next(generator)
  except (StopIteration, errors.OutOfRangeError):
    return None

  if not isinstance(generator_output, tuple):
    # Always wrap in a tuple.
    generator_output = (generator_output,)

  if len(generator_output) not in [1, 2, 3]:
    raise ValueError(
        'Output of generator should be a tuple of 1 or 2 or 3 '
        'elements: (input,) or (input, target) or '
        '(input, target, sample_weights). Received {}'.format(generator_output))
  return generator_output

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
        num_batches = 0
        #
        # for features_n_label in next(gen):
        #     for j in range(len(features_n_label)):
        #         if num_batches < 2:
        #             for i in range(len(features_n_label[j])):
        #                 print(features_n_label[j][i].shape)
        #     # print("\n")
        #     num_batches += 1
        #     print(num_batches)


        for features_n_label in _get_next_batch(gen):
            for j in range(len(features_n_label)):
                if num_batches < 2:
                    for i in range(len(features_n_label[j])):
                        print(features_n_label[j][i].shape)
            # print("\n")
            num_batches += 1
            print(num_batches)

if __name__ == '__main__':
    unittest.main()