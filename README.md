[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/imaginea/vitaflow/blob/master/LICENSE)


# VitaFlow - VideoImageTextAudioFlow
 ![](vitaflow-logo.png)

## [Introduction](what_is_vitaflow.md)

## Environment Setup

**Python Setup**

```
   git clone https://github.com/Imaginea/vitaFlow/
   cd vitaFlow/
   conda create -n vf python=3.5
   conda activate vf
   export SLUGIFY_USES_TEXT_UNIDECODE=yes
   pip install -r requirements.txt
```

**Ubuntu Specific Installation**
- https://www.tensorflow.org/tfx/serving/setup
- `sudo apt-get -y install postgresql postgresql-contrib libpq-dev postgresql-client postgresql-client-common`


---------------------------------------------------------------------------------------------------------

## Demo  
We have put up a small working demo, which basically can read text from camera images. (Note: the models are not tweaked for the maximum performance)   

The pipeline components includes:   
 - EAST Model
 - Image Binarization 
 - Cropping tool (crops each text line from the image as single image)
 - OCR with Tesseract and DL based OCR called Calamari  (image to text for all the images that were generated from previous cropper stage)
 - Text stiching (where each text out from the images are stiched as one text file)
 
### Local machine

**MakeFile**  

- Input Files Directory : [data/receipts/](data/receipts/)
- output  Files Directory : [data/text_out/](data/text_out/)
  - X51008142068_**CalamariOcrPlugin**.txt : **Output using our pipeline**
  - X51008142068_**TessaractOcrPlugin**.txt : **Ouput using PyTesseract**
  
```
   make east_ocr_pipeline
```

**Web UI**
```
   cd path/to/vitaflow/
   cd vitaflow/annotate_server/
   python ./demo.py
```
Access the Web UI @ http://127.0.0.1:5000/

Home Page UI:

![](docs/images/vitaflow_annotation_tool_demo_start.png)

Sample out of a Image:

![](docs/images/vitaflow_annotation_tool_demo_end.png)


**Airflow**

![](docs/images/east_airflow.png)

```
   export AIRFLOW_HOME=~/airflow # set airflow config dir (default : ~/airflow)
   mkdir $AIRFLOW_HOME

   airflow version # init cnf files
   airflow initdb # init sqlite file

   sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' $AIRFLOW_HOME/airflow.cfg
   sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
   sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
   sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' $AIRFLOW_HOME/airflow.cfg
   sed -i'.orig' 's/load_examples = True/load_examples = False/g' $AIRFLOW_HOME/airflow.cfg
    
    #create Posgresql user and DB for Airflow metadata store
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
    \q
    
    
    sudo vim /etc/postgresql/10/main/pg_hba.conf
    # IPv4 local connections:
    host    all             all             0.0.0.0/0               md5
    
    sudo vim /etc/postgresql/10/main/postgresql.conf
    listen_addresses = '*'
    
    sudo service postgresql restart
    
    vim $AIRFLOW_HOME/airflow.cfg
    # update below 4 parameters
    executor = LocalExecutor
    sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
    
    broker_url = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
    celery_result_backend = postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
    
    airflow resetdb --yes
    airflow initdb
  
```

```
#following make command will copy the entire vitaflow python code base to $AIRFLOW_HOME/dags folder
export AIRFLOW_HOME=~/airflow
make airflow
```

On a new shell:
```
export AIRFLOW_HOME=~/airflow
airflow webserver
```

On a new shell:
```
export AIRFLOW_HOME=~/airflow
airflow scheduler
```

Airflow UI @ http://0.0.0.0:8080/admin/
 - Turn on the `east_flow` DAG
 - Click `east_flow` DAG
 - On the UI click `Trigger DAG`
 
Data:
- Check [vitaflow/pipeline/east_airflow_demo/east_flow.py](vitaflow/pipeline/east_airflow_demo/east_flow.py) for data folder configuration detatils
- In general for Airflow **/tmp/vitaflow/** is used

**Note:** It is jus a POC to use the pipeline as part of Airflow. Stay tuned for improvements.

### Docker

- TODO 

---------------------------------------------------------------------------------------------------------

## Developers

How to tune EAST model and Calamari model for better OCR.

## Training

### [EAST](vitaflow/playground/east/README.md)

### Calamari

- TODO

##### Disclaimer : VitaFlow is in it's infant stage! Feel free to review/suggestion/feedbacks!
