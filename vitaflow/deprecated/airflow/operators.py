from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from vitaflow.pipeline.preprocessor.binarisation import ImageBinarisePreprocessor
from vitaflow.pipeline.preprocessor.crop_to_box import EastCropperModule

from vitaflow.pipeline.postprocessor.ocr_tesseract import TessaractOcrModule
from vitaflow.pipeline.postprocessor.ocr_calamari import CalamariOcrModule
from vitaflow.pipeline.postprocessor.text_file_stitch import TextFile

class ImageBinariseOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        BaseOperator.__init__(self, *args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        binarization = ImageBinarisePreprocessor()
        binarization.process_files(source_dir=self.source_folder,
                                   destination_dir=self.destination_folder)


class EastCropperImageOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder', 'east_out_dir')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 east_out_dir,
                 *args, **kwargs):
        BaseOperator.__init__(self, *args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.east_out_dir = east_out_dir

    def execute(self, context):
        cropper = EastCropperModule(east_out_dir=self.east_out_dir)
        cropper.process_files(source_dir=self.source_folder,
                              destination_dir=self.destination_folder)


class TessaractOcrOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        BaseOperator.__init__(self, *args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        ocr = TessaractOcrModule()
        ocr.process_files(source_dir=self.source_folder,
                          destination_dir=self.destination_folder)


class CalamariOcrOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        BaseOperator.__init__(self, *args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        ocr = CalamariOcrModule()
        ocr.process_files(source_dir=self.source_folder,
                          destination_dir=self.destination_folder)


class TextCombinerOperator(BaseOperator):
    template_fields = ('source_folder', 'destination_folder')
    ui_color = '#A6E6A6'

    @apply_defaults
    def __init__(self,
                 source_folder,
                 destination_folder,
                 *args, **kwargs):
        BaseOperator.__init__(self, *args, **kwargs)
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def execute(self, context):
        aggregator = TextFile()
        aggregator.process_files(source_dir=self.source_folder,
                                 destination_dir=self.destination_folder)
