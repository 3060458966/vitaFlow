'''
Demo sample example of how to include calamari_ocr into python code
'''
from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import run


# TODO:
# - convert to gray scale images
# - fixed width input 42 px


class args:
    batch_size = 1
    checkpoint = ['/Users/sampathm/model_00131400.ckpt']  # Add your files here
    dataset = DataSetType.FILE
    extended_prediction_data = False
    extended_prediction_data_format = 'json'
    files = ['/Users/sampathm/10005.jpg',
             '/Users/sampathm/10006.jpg',
             '/Users/sampathm/10504.jpg',
             '/Users/sampathm/new2.png',
             '/Users/sampathm/X51008123447.jpg']  # Add your files here
    no_progress_bars = False
    output_dir = None
    pagexml_text_index = 1
    processes = 1
    text_files = None
    verbose = False
    voter = 'confidence_voter_default_ctc'


run(args)
