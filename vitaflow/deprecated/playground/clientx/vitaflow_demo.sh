#!/usr/bin/env bash
export DEMO_DATA_PATH=/opt/data/vitaflow_demo
cd /opt/github/vitaFlow/

# os.environ['DEMO_DATA_PATH']
cd /opt/github/vitaFlow/vitaflow/image_synthesizer/
#vim plain_text_image.py
python plain_text_image.py

cd /opt/github/vitaFlow/examples/clientx/standoff2conll/
python2 standoff2conll.py $DEMO_DATA_PATH/receipt_mock_data/train/
mkdir $DEMO_DATA_PATH/train/
mv csv/* $DEMO_DATA_PATH/train/

#python2 standoff2conll.py $DEMO_DATA_PATH/receipt_mock_data/test/
mkdir $DEMO_DATA_PATH/test/
#mv csv/* $DEMO_DATA_PATH/test/

python2 standoff2conll.py $DEMO_DATA_PATH/receipt_mock_data/val/
mkdir $DEMO_DATA_PATH/val/
mv csv/* $DEMO_DATA_PATH/val/



cd /opt/github/vitaFlow/
#vim examples/clientx/config.py
#vim examples/clientx/clientx_dataset.py
python vitaflow/bin/vf_engine.py --mode=train --config_python_file=examples/clientx/config.py

python vitaflow/app/main.py 


scp -r /opt/data/vitaflow_demo/receipt_mock_data/train/ hadoop-user@192.168.2.217:/home/hadoop-user/brat/data/mages/
scp -r /opt/data/vitaflow_demo/receipt_mock_data/test/ hadoop-user@192.168.2.217:/home/hadoop-user/brat/data/mages/
scp -r /opt/data/vitaflow_demo/receipt_mock_data/val/ hadoop-user@192.168.2.217:/home/hadoop-user/brat/data/val/
http://192.168.2.217:8002/index.xhtml#/
