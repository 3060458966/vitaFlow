from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from vitaflow.pipeline.preprocessor.binarisation import ImageBinarisePreprocessor
from vitaflow.pipeline.preprocessor.crop_to_box import EastCropperImagePlugin

class ImageBinarisePreprocessorOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        super(ImageBinarisePreprocessorOperator, self).__init__(*args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        binarization = ImageBinarisePreprocessor()
        binarization.process_files(source_dir=self.source_folder,
                                   destination_dir=self.destination_folder)


class EastCropperImagePreprocessorOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        super(ImageBinarisePreprocessorOperator, self).__init__(*args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        cropper = EastCropperImagePlugin()
        binarization.process_files(source_dir=self.source_folder,
                                   destination_dir=self.destination_folder)
