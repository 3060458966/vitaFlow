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




from vitaflow.pipeline.airflow.preprocessor import ImageBinarisePreprocessorOperator

input_dir="/opt/github/vitaFlow/vitaflow/annotate_server/static/data/preprocess"
output_dir="/opt/tmp/vitaFlow/east_airflow_demo/"
model_dir="/opt/github/vitaFlow/vitaflow/annotate_server/static/data/east_models/east_airflow_demo/EASTModel/exported/1558013588"


default_args = {
    'owner': 'east_flow',
    'depends_on_past': False,
    'start_date': datetime(2019, 4, 29),
    'provide_context': True,
    'retries': 100,
    'retry_delay': timedelta(seconds=30)
}

task_name = 'east_flow1'
# try:
#     gin.parse_config_file('vitaflow/pipeline/east_airflow_demo/east_flow_config.gin')
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
binarization = ImageBinarisePreprocessorOperator(task_id="binarization",
                                                 source_folder=output_dir,
                                                 destination_folder="/opt/tmp/vitaFlow/east_binarized/")
end = DummyOperator(task_id='end', dag=dag)

#===========================================================================================

start.set_downstream(prediction_task)

prediction_task.set_downstream(binarization)

binarization.set_downstream(end)
