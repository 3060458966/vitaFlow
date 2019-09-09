from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import generics
from .serializers import ImageUploadSerializer
from rest_framework import status

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
            obj = serializer.save()
            saved_file_name = str(obj).split('/')[-1]
        else:
            return Response({serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"{} - Uploaded Successfully".format(saved_file_name)}, status=status.HTTP_200_OK)

class ProcessImage(generics.GenericAPIView):

    def get(self, request, *args, **kwargs):
        """
        trigger the vitaflow functions
        and return 200 if successful
        :return:
        """
        pass

class GetProcessedImage(generics.RetrieveAPIView):

    def get(self, request, *args, **kwargs):
        pass