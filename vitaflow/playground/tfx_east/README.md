
## Getting Started
 1. Create a new environment.
 2. Install dependencies
 3. Clone the repository 

```
    # Pyhton 2.7
    cd
    virtualenv -p python2.7 tfx-env
    source ~/tfx-env/bin/activate
    mkdir tfx; cd tfx
    
    pip install tensorflow==1.13.1
    pip install tfx==0.12.0
    git clone https://github.com/tensorflow/tfx.git
    cd ~/tfx/tfx/tfx/examples/workshop/setup
    ./setup_demo.sh
    
    # Python 3.5
    pip install tensorflow_gpu
    pip uninstall enum34
    pip install "apache-airflow[mysql, postgresql, celery, rabbitmq]"
    
    #kombu 4.5.0 has requirement amqp<3.0,>=2.4.0, but you'll have amqp 1.4.9 which is incompatible.
    
    export AIRFLOW_HOME=~/airflow # set airflow config dir (default : ~/airflow)
    mkdir $AIRFLOW_HOME
  
    airflow version # init cnf files
    airflow initdb # init sqlite file

    sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' $AIRFLOW_HOME/airflow.cfg
    sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
    sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
    sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' $AIRFLOW_HOME/airflow.cfg
    sed -i'.orig' 's/load_examples = True/load_examples = False/g' $AIRFLOW_HOME/airflow.cfg

    airflow resetdb --yes
    airflow initdb
  
    cd /path/to/tfx/
    pip install setup.py --user
    jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis
    jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis
    
    cd tfx/examples/workshop/
  
```


```
# Airflow + Posgres
# https://vujade.co/install-apache-airflow-ubuntu-18-04/
# https://gist.github.com/zacgca/9e0401aa205e7c54cbae0e85afca479d
# https://gist.github.com/rosiehoyem/9e111067fe4373eb701daf9e7abcc423

#posgres url 
postgresql://user:password@localhost:5432/database_name

sudo -u postgres psql
CREATE ROLE airflow WITH
  LOGIN
  SUPERUSER
  INHERIT
  CREATEDB
  CREATEROLE
  REPLICATION;
  
CREATE ROLE airflow;
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES on database airflow to airflow;
ALTER ROLE airflow SUPERUSER;
ALTER ROLE airflow CREATEDB;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public to airflow;
\password airflow
ALTER ROLE "airflow" WITH LOGIN;

\c airflow
\conninfo 


sudo vim /etc/postgresql/10/main/pg_hba.conf
# IPv4 local connections:
host    all             all             0.0.0.0/0               md5

sudo vim /etc/postgresql/10/main/postgresql.conf
listen_addresses = '*'

sudo service postgresql restart

#posgres url 
postgresql://airflow:airflow@localhost:5432/airflow

executor = CeleryExecutor
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow

broker_url = amqp://guest:guest@localhost:5672//
celery_result_backend = amqp://guest:guest@localhost:5672//

broker_url = db+postgresql://airflow:airflow@localhost:5432/airflow
celery_result_backend = db+postgresql://airflow:airflow@localhost:5432/airflow

broker_url = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
celery_result_backend = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow

```
setup_demo will install the remaining dependencies and setup other requisites.

This will create a folder called *Airflow* in your home directory.

## Adding Data
1. Change the directory to ~/airflow ($AIRFLOW_HOME)
2. Create a folder called east under the data folder 
3. Copy the tf record files into folder called east.

## Adding Dags

1. Delete the contents of the dag folder. 
2. Copy the contents from the tfx_east into the dag folder

## Running Code
1. Getting the dag to start requires starting the webserver and scheduler component to be triggered.


```
    # Open a new terminal window, and in that window ...
    source ~/tfx-env/bin/activate
    airflow webserver -p 8080
    
    # Open another new terminal window, and in that window ...
    source ~/tfx-env/bin/activate
    airflow scheduler
    
    or

    #in one terminal run the web service:
    export AIRFLOW_HOME=~/airflow
    airflow webserver --port=8080 # web ui
    
    #in the other terminal:
    export AIRFLOW_HOME=~/airflow
    airflow scheduler # actual executer
    
```
In a browser:

    Open a browser and go to http://127.0.0.1:8080

1. Graph will be listed in the grid view. 
2. Turn the toggle button to get the graph in ON state.
3. Press *trigger graph button*. 

This will start the processing of the graph.

## References

1.https://www.tensorflow.org/tfx/tutorials/tfx/workshop
