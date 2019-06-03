# import datetime as dt
# from airflow import DAG
# from airflow.operators.bash_operator import BashOperator
# from airflow.operators.python_operator import PythonOperator
#
# default_args = {
#     'owner': 'imaginea',
#     'start_date': dt.datetime(2017, 6, 1),
#     'retries': 1,
#     'retry_delay': dt.timedelta(minutes=5),
# }
#
#
# with DAG('airflow_tutorial_v01',
#          default_args=default_args,
#          schedule_interval='0 0 * * *', #https://crontab.guru/#0_*_*_*_*
#          ) as dag:
#
#     def print_world():
#         print('world')
#
#     print_hello = BashOperator(task_id='print_hello',
#                                bash_command='echo "hello"')
#     sleep = BashOperator(task_id='sleep',
#                          bash_command='sleep 5')
#     print_world = PythonOperator(task_id='print_world',
#                                  python_callable=print_world)
#
# print_hello >> sleep >> print_world