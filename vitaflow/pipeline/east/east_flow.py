from airflow import DAG
# from airflow.operators import PythonOperator, OmegaFileSensor, ArchiveFileOperator, TriggerDagRunOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime, timedelta
from airflow.models import Variable
from vitaflow.playground.east.pb_predict import get_text_segmentation_pb
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import argparse
import os
import gin

from tensorflow.contrib import predictor

try:
    from grpc_predict import read_image, get_text_segmentation_pb
    from icdar_data import get_images
except:
    from vitaflow.playground.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.playground.east.icdar_data import get_images


input_dir="/opt/github/vitaFlow/vitaflow/annotate_server/static/data/preprocess"
output_dir="/opt/tmp/vitaFlow/east/"
model_dir="/opt/github/vitaFlow/vitaflow/annotate_server/static/data/east_models/east/EASTModel/exported/1558013588"

# @gin.configurable
def east_flow_predictions(input_dir=input_dir, output_dir=output_dir, model_dir=model_dir):
    images_dir = input_dir
    images = get_images(images_dir)
    predict_fn = predictor.from_saved_model(model_dir)
    for image_file_path in images:
        im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
        result = predict_fn({'images': img_resized})
        get_text_segmentation_pb(img_mat=im,
                                 result=result,
                                 output_dir=output_dir,
                                 file_name=os.path.basename(image_file_path),
                                 ratio_h=ratio_h,
                                 ratio_w=ratio_w)


default_args = {
    'owner': 'east_flow',
    'depends_on_past': False,
    'start_date': datetime(2019, 4, 29),
    'provide_context': True,
    'retries': 100,
    'retry_delay': timedelta(seconds=30)
}

task_name = 'east_flow'
# try:
#     gin.parse_config_file('vitaflow/pipeline/east/east_flow_config.gin')
# except:
#     gin.parse_config_file('east_flow_config.gin')

# with DAG(task_name,
#         start_date=datetime(2019, 4, 29),
#         default_args=default_args,
#         schedule_interval=None,
#         catchup=False,
#         max_active_runs=1,
#         concurrency=1) as dag:

dag = DAG(task_name,
             start_date=datetime(2019, 4, 29),
             default_args=default_args,
             schedule_interval='0 0 * * *', #https://crontab.guru/#0_*_*_*_*,
             catchup=False,
             max_active_runs=1,
             concurrency=1)

start = DummyOperator(task_id='start', dag=dag)

prediction_task = PythonOperator(task_id='east_predictions',
                                 python_callable=east_flow_predictions)

end = DummyOperator(task_id='end', dag=dag)

start.set_downstream(prediction_task)
prediction_task.set_downstream(end)
