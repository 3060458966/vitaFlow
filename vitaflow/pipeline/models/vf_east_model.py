import sys
import os
sys.path.append(os.getcwd())
import fire


try:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images
except:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images

# from tensorflow.contrib.predictor
import tensorflow as tf
from vitaflow.deprecated import demo_config


def east_flow_predictions(input_dir=demo_config.IMAGE_ROOT_DIR,
                          output_dir=demo_config.EAST_OUT_DIR,
                          model_dir=demo_config.EAST_MODEL_DIR):
    print(">>>>>>>>>>>>>>>>>>>>>", input_dir)
    images_dir = input_dir
    images = get_images(images_dir)
    predict_fn = tf.contrib.predictor.from_saved_model(model_dir)
    for image_file_path in images:
        im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
        result = predict_fn({'images': img_resized})
        get_text_segmentation_pb(img_mat=im,
                                 result=result,
                                 output_dir=output_dir,
                                 file_name=os.path.basename(image_file_path),
                                 ratio_h=ratio_h,
                                 ratio_w=ratio_w)


def run(input_dir,
        output_dir,
        model_dir):
    """
    Predict script for EAST model
    :param input_dir: Directory containing images
    :param output_dir: Directory to store the predictions as text file and images (with text boxes)
    :param model_dir: EAST model directory
    :return:
    """

    east_flow_predictions(input_dir=args.images_dir)


if __name__ == '__main__':
    fire.Fire(run)

