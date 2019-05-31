from airflow.models import BaseOperator

from vitaflow.pipeline.preprocessor.binarisation import ImageBinarisePreprocessor

class ImageBinarisePreprocessorOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(
            self,
            source_folder,
            destination_folder,
            *args, **kwargs):
        # relative path
        self.source_folder = source_folder
        # relative path
        self.destination_folder = destination_folder

    def execute(self, context):
        binarization = ImageBinarisePreprocessor(source_folder=self.source_folder,
                                                 destination_folder=self.destination_folder)


