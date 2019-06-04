from airflow import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime, timedelta
from airflow.operators.dummy_operator import DummyOperator
import os

from vitaflow.pipeline.airflow.models import EastModelOperator
from vitaflow.pipeline.airflow.operators import ImageBinariseOperator, EastCropperImageOperator, TessaractOcrOperator, CalamariOcrOperator
from vitaflow.pipeline.airflow.operators import TextCombinerOperator
from vitaflow.pipeline.interfaces.utils import check_n_create


IMAGE_ROOT_DIR = "/tmp/vitaflow/receipts/"
EAST_MODEL_DIR = "vitaflow/annotate_server/static/data/east_models/east/EASTModel/exported/1558013588"
EAST_OUT_DIR = "/tmp/vitaflow/east"
BINARIZE_ROOT_DIR = "/tmp/vitaflow/binarisation"
CROPPER_ROOT_DIR = "/tmp/vitaflow/cropped_images"
TEXT_OCR_DATA_DIR = "/tmp/vitaflow/text_data"
TEXT_OUT_DIR = "/tmp/vitaflow/text_out"
LOGS_DIR = "/tmp/vitaflow/logs/"

def create_dirs():
    # create missing dir
    for each_dir in [IMAGE_ROOT_DIR,
                     EAST_OUT_DIR,
                     CROPPER_ROOT_DIR,
                     BINARIZE_ROOT_DIR,
                     TEXT_OCR_DATA_DIR,
                     TEXT_OUT_DIR,
                     LOGS_DIR]:
        each_dir = os.path.join(each_dir)
        print(each_dir)
        check_n_create(each_dir)


default_args = {
    'owner': 'east_flow',
    'depends_on_past': False,
    'start_date': datetime(2019, 4, 29),
    'provide_context': True,
    'retries': 100,
    'retry_delay': timedelta(seconds=30)
}

task_name = 'east_flow1'

dag = DAG(task_name,
          start_date=datetime(2019, 4, 29),
          default_args=default_args,
          schedule_interval='0 0 * * *', #https://crontab.guru/#0_*_*_*_*,
          catchup=False,
          max_active_runs=1,
          concurrency=1)

start = DummyOperator(task_id='start', dag=dag)

prediction_task = EastModelOperator(task_id="east",
                                    source_folder=IMAGE_ROOT_DIR,
                                    destination_folder=EAST_OUT_DIR,
                                    model_dir=EAST_MODEL_DIR)

binarization = ImageBinariseOperator(task_id="binarization",
                                     source_folder=EAST_OUT_DIR,
                                     destination_folder=BINARIZE_ROOT_DIR)

cropper = EastCropperImageOperator(task_id="cropper",
                                   east_out_dir=EAST_OUT_DIR,
                                   source_folder=BINARIZE_ROOT_DIR,
                                   destination_folder=CROPPER_ROOT_DIR)

tesseract = TessaractOcrOperator(task_id="tesseract",
                                 source_folder=CROPPER_ROOT_DIR,
                                 destination_folder=TEXT_OCR_DATA_DIR)

calamari = CalamariOcrOperator(task_id="calamari",
                               source_folder=CROPPER_ROOT_DIR,
                               destination_folder=TEXT_OCR_DATA_DIR)

text_stiching = TextCombinerOperator(task_id="text_stiching",
                                     source_folder=TEXT_OCR_DATA_DIR,
                                     destination_folder=TEXT_OUT_DIR)

end = DummyOperator(task_id='end', dag=dag)

#===========================================================================================

start.set_downstream(prediction_task)

prediction_task.set_downstream(binarization)

binarization.set_downstream(cropper)

cropper.set_downstream([tesseract, calamari])

tesseract.set_downstream(text_stiching)
calamari.set_downstream(text_stiching)

text_stiching.set_downstream(end)


