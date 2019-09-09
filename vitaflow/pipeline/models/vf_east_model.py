import os
import argparse


try:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images
except:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images

from tensorflow.contrib import predictor
from vitaflow import demo_config

from vitaflow.demo_config import create_dirs


def east_flow_predictions(input_dir=demo_config.IMAGE_ROOT_DIR,
                          output_dir=demo_config.EAST_OUT_DIR,
                          model_dir=demo_config.EAST_MODEL_DIR):
    print(">>>>>>>>>>>>>>>>>>>>>", input_dir)
    images_dir = input_dir
    images = get_images(images_dir)
    predict_fn = predictor.from_saved_model(model_dir)
    for image_file_path in images:
        im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
        result = predict_fn({'images': img_resized})
        get_text_segmentation_pb(img_mat=im,
                                 result=result,
                                 output_dir=output_dir,
                                 file_name=os.path.basename(image_file_path),
                                 ratio_h=ratio_h,
                                 ratio_w=ratio_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default=demo_config.IMAGE_ROOT_DIR, help='input images', type=str)
    create_dirs()
    args = parser.parse_args()
    east_flow_predictions(input_dir=args.images_dir)
