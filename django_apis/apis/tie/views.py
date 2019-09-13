import os
import shutil
import sys
sys.path.append('/opt/vlab/vitaFlow')

from vitaflow.pipeline.postprocessor.ocr_calamari import CalamariOcrPlugin
from vitaflow.pipeline.postprocessor.ocr_tesseract import TessaractOcrPlugin
from vitaflow.pipeline.postprocessor.text_file_stitch import TextFile
from vitaflow.pipeline.preprocessor.binarisation import ImageBinarisePreprocessor
from vitaflow.pipeline.preprocessor.crop_to_box import EastCropperImagePlugin



from django.shortcuts import render
from rest_framework.response import Response

from rest_framework import generics

from vitaflow.pipeline.models.vf_east_model import east_flow_predictions
from .serializers import ImageUploadSerializer
from rest_framework import status
from django.conf import settings

import base64


def image_as_base64(image_file, format='jpg'):
    """
    :param `image_file` for the complete path of image.
    :param `format` is format for image, eg: `png` or `jpg`.
    """
    if not os.path.isfile(image_file):
        return None

    encoded_string = ''
    with open(image_file, 'rb') as img_f:
        encoded_string = base64.b64encode(img_f.read())
    return 'data:image/%s;base64,%s' % (format, encoded_string)

# Create your views here.

class UploadImage(generics.CreateAPIView):

    serializer_class = ImageUploadSerializer

    def post(self, request, *args, **kwargs):
        """
        get the image from user , store in dir with name provided or with name = imgae file name
        return stored file name
        :return:
        """
        print(request.data)
        serializer = self.get_serializer(data = request.data)
        if serializer.is_valid():
            print("Deleting old EAST directories....")
            if os.path.exists(settings.IMG_DIR):
                shutil.rmtree(settings.IMG_DIR)
                os.makedirs(settings.IMG_DIR)
            if os.path.exists(settings.EAST_OUT_IMG_DIR):
                shutil.rmtree(settings.EAST_OUT_IMG_DIR)
                os.makedirs(settings.EAST_OUT_IMG_DIR)

            obj = serializer.save()
            saved_file_name = str(obj).split('/')[-1]
            print("Sending Response")
            return Response({'img_file': saved_file_name}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        print("2 sending response")
        return Response({'img_file': saved_file_name}, status=status.HTTP_200_OK)

class ProcessImage(generics.GenericAPIView):

    def get(self, request, *args, **kwargs):
        """
        trigger the vitaflow functions
        and return 200 if successful
        :return:
        """
        east_flow_predictions(input_dir=settings.IMG_DIR,
                              output_dir=settings.EAST_OUT_IMG_DIR,
                              model_dir=settings.EAST_MODEL_DIR)
        return Response({"SUCCESS"}, status=status.HTTP_200_OK)


class GetTextLocalization(generics.GenericAPIView):

    def get(self, request, *args, **kwargs):
        file_name = request.query_params.get('file_name', None)
        url = os.path.join(os.path.join(settings.BASE_DIR, settings.EAST_OUT_IMG_DIR), file_name)
        if os.path.exists(url):
            with open(url, 'rb') as img_f:
                encoded_string = base64.b64encode(img_f.read())
            return Response({'localised_image': encoded_string})
        else:
            return Response({'err':"{} File not Found".format(url)}, status=status.HTTP_400_BAD_REQUEST)

class GetLocalizedText(generics.RetrieveAPIView):

    def get(self, request, *args, **kwargs):
        t = ImageBinarisePreprocessor(weights_path=settings.BINARIZER_MODEL_WEIGTHS)
        print('--' * 55)
        t.process_files(source_dir=settings.EAST_OUT_IMG_DIR, destination_dir=settings.BINARIZE_ROOT_DIR)

        t = EastCropperImagePlugin(east_out_dir=settings.EAST_OUT_IMG_DIR)
        print('--' * 55)
        t.process_files(source_dir=settings.BINARIZE_ROOT_DIR, destination_dir=settings.CROPPER_ROOT_DIR)

        tt = TessaractOcrPlugin(num_workers=4)
        print('--' * 55)
        tt.process_files(source_dir=settings.CROPPER_ROOT_DIR,
                         destination_dir=settings.TEXT_OCR_DATA_DIR)

        calamari = CalamariOcrPlugin()

        calamari.process_files(source_dir=settings.CROPPER_ROOT_DIR,
                               destination_dir=settings.TEXT_OCR_DATA_DIR,
                               keep_destination=True)

        tt = TextFile()
        print('--' * 55)
        extracted_text = tt.process_files(source_dir=settings.TEXT_OCR_DATA_DIR, destination_dir=settings.TEXT_OUT_DIR)

        print("="*50)
        print(extracted_text)

        return Response(extracted_text)


