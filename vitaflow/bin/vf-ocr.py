import os
import shutil
import argparse
import sys
sys.path.append('/opt/vlab/vitaFlow')

from vitaflow.pipeline.models.vf_east_model import east_flow_predictions



from vitaflow.pipeline.postprocessor.ocr_calamari import CalamariOcrPlugin
from vitaflow.pipeline.postprocessor.ocr_tesseract import TessaractOcrPlugin
from vitaflow.pipeline.postprocessor.text_file_stitch import TextFile
from vitaflow.pipeline.preprocessor.binarisation import ImageBinarisePreprocessor
from vitaflow.pipeline.preprocessor.crop_to_box import EastCropperImagePlugin


def main(args):
    our_dir = args['out_dir']
    EAST_MODEL_DIR = args['east_model_dir']
    CALAMARI_MODEL_DIR = [args['calamari_model_dir'] + '/model_00117200.ckpt',
                          args['calamari_model_dir'] + '/model_00132600.ckpt']
    EAST_OUT_IMG_DIR = our_dir + '/gen/east_out'
    BINARIZE_ROOT_DIR = our_dir + '/gen/binarized_data'
    CROPPER_ROOT_DIR = our_dir + '/gen/cropped_data'
    TEXT_OCR_DATA_DIR = our_dir + '/gen/ocr_text_data'
    TEXT_OUT_DIR = our_dir + '/gen/final_text'
    BINARIZER_MODEL_WEIGTHS = args['binarizer_weights_path']

    east_flow_predictions(input_dir=args['image_dir'],
                          output_dir=EAST_OUT_IMG_DIR,
                          model_dir=EAST_MODEL_DIR)

    t = ImageBinarisePreprocessor(weights_path=BINARIZER_MODEL_WEIGTHS)
    print('--' * 55)
    t.process_files(source_dir=EAST_OUT_IMG_DIR, destination_dir=BINARIZE_ROOT_DIR)

    t = EastCropperImagePlugin(east_out_dir=EAST_OUT_IMG_DIR)
    print('--' * 55)
    t.process_files(source_dir=BINARIZE_ROOT_DIR, destination_dir=CROPPER_ROOT_DIR)

    tt = TessaractOcrPlugin(num_workers=4)
    print('--' * 55)
    tt.process_files(source_dir=CROPPER_ROOT_DIR,
                     destination_dir=TEXT_OCR_DATA_DIR)

    calamari = CalamariOcrPlugin(calamari_models=CALAMARI_MODEL_DIR)
    calamari.process_files(source_dir=CROPPER_ROOT_DIR,
                           destination_dir=TEXT_OCR_DATA_DIR,
                           keep_destination=True)

    tt = TextFile()
    print('--' * 55)
    extracted_text = tt.process_files(source_dir=TEXT_OCR_DATA_DIR, destination_dir=TEXT_OUT_DIR)

    print("=" * 50)
    print(extracted_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OCR (EAST+Calamari)')

    parser.add_argument('-id', "--image_dir", type=str, required=True, help="Image Directory to run the OCR")
    parser.add_argument('-od', "--out_dir", type=str, required=True, help="Image out directory to run the OCR")

    parser.add_argument('-em', "--east_model_dir", type=str, default="django_apis/apis/data/models/vf_east_models/east/EASTModel/exported/1558013588/",
                        help="Path to EAST Model directory")
    parser.add_argument('-cm', "--calamari_model_dir", type=str, default="django_apis/apis/data/models/calamari_models/", help="Path to Calamari Model directory")
    parser.add_argument('-bp', "--binarizer_weights_path", type=str, default="django_apis/apis/data/models/binarizer/bin_weights.hdf5", help="Path to Calamari Model directory")


    args = vars(parser.parse_args())

    main(args)