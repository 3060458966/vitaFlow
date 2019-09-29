from glob import glob

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from glob import glob

from vitaflow.utils.print_helper import print_info
from tqdm import tqdm

from vitaflow.helpers.print_helper import print_info


def image_annotations(path_to_tensorflow_model, category_index, images_src, images_dest):
    def get_box_dims(box, image_shape):
        ymin, xmin, ymax, xmax = box
        im_width, im_height, im_depth = image_shape
        ymin, xmin, ymax, xmax = map(int, (xmin * im_width, xmax * im_width,
                                           ymin * im_height, ymax * im_height))
        return (ymax, xmax, ymin, xmin)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_tensorflow_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    bag = []

    for image_path in tqdm(glob(images_src+"/*")):
        print_info("Processing {}".format(image_path))
        image = plt.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        mask = scores > 0.3

        image_shape = image.shape

        coords = list(map(lambda x: get_box_dims(x, image_shape), boxes[mask].tolist()))
        tags = list(map(lambda x: category_index[int(x)]['name'], classes[mask].tolist()))
        scores = scores[mask].tolist()
        bag.append({'image_loc': image_path,
                    'dest': images_dest,
                    'coords': coords,
                    'tags': tags,
                    'scores': scores})

    # pprint(bag)
    return bag

