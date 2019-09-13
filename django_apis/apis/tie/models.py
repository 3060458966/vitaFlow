from django.db import models
from django.conf import settings
from utils import utils

class ImageDetails(models.Model):
    image = models.ImageField(db_column='image', upload_to=utils.rename_and_upload_path)
    image_name = models.CharField(db_column='image_name', max_length=255)

    def __str__(self):
        return str(self.image)