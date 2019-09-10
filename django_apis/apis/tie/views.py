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

# Create your views here.

class UploadImage(generics.CreateAPIView):

    serializer_class = ImageUploadSerializer

    def post(self, request, *args, **kwargs):
        """
        get the image from user , store in dir with name provided or with name = imgae file name
        return stored file name
        :return:
        """
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
        else:
            return Response({serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
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
        url = os.path.join(settings.EAST_OUT_IMG_DIR, file_name)
        return Response({'localised_image': url})


class GetLocalizedText(generics.RetrieveAPIView):

    def get(self, request, *args, **kwargs):
        t = ImageBinarisePreprocessor()
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


