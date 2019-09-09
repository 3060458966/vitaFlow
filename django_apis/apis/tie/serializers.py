from rest_framework import serializers
from .models import ImageDetails


class ImageUploadSerializer(serializers.ModelSerializer):

    class Meta :
        model = ImageDetails
        fields = ('image', )