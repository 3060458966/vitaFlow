'''
Demo sample example of how to include calamari_ocr into python code
'''

import os

from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import run as calamari_ocr_run

# import config
from vitaflow.pipeline.interfaces.plugin import OCRPluginInterface

from vitaflow import demo_config

# TODO:
# - convert to gray scale images
# - fixed width input 42 px


# Add your files here
calamari_models = ['data/models/calamari_models/model_00117200.ckpt',
                   'data/models/calamari_models/model_00132600.ckpt']

calamari_input_images = []  # glob(os.path.join(config.ROOT_DIR, config.TEXT_IMAGES) + '/*/*')  # Add your files here


class CalamariArgs:
    batch_size = 1
    checkpoint = calamari_models
    dataset = DataSetType.FILE
    extended_prediction_data = False
    extended_prediction_data_format = 'json'
    files = calamari_input_images
    no_progress_bars = False
    output_dir = None
    pagexml_text_index = 1
    processes = 1
    text_files = None
    verbose = False
    voter = 'confidence_voter_default_ctc'
    extension = None


class CalamariOcrPlugin(OCRPluginInterface):

    def __init__(self,
                 calamari_models=None,
                 num_workers=4):
        OCRPluginInterface.__init__(self,
                                    num_workers=num_workers)
        self._calamari_models = calamari_models

    def _handle_file(self, in_file_path, out_file_path):
        destination_dir = out_file_path.split("/")[-2]
        CalamariArgs.files = [in_file_path]
        CalamariArgs.output_dir = destination_dir
        calamari_ocr_run(CalamariArgs)

    def _handle_files(self, source_dir, destination_dir):
        """Plugin module should implement this to handle all the files in the given directory"""

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        if self._calamari_models is not None:
            CalamariArgs.checkpoint = self._calamari_models

        for dir in os.listdir(source_dir):

            in_files = self.get_all_input_files(source_dir=os.path.join(source_dir, dir))

            if in_files:
                output_dir = os.path.join(destination_dir, dir)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                CalamariArgs.files = in_files
                CalamariArgs.output_dir = output_dir
                CalamariArgs.batch_size = len(in_files)
                CalamariArgs.processes = len(in_files)
                calamari_ocr_run(CalamariArgs)


if __name__ == '__main__':
    calamari = CalamariOcrPlugin()
    calamari.process_files(source_dir=demo_config.CROPPER_ROOT_DIR,
                           destination_dir=demo_config.TEXT_OCR_DATA_DIR)
