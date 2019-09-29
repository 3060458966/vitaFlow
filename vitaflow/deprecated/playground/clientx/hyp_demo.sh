#!/usr/bin/env bash
# Data path
#export DEMO_DATA_PATH=/opt/data/vitaflow_demo
# trial run keep the above demo_data_path

export DEMO_DATA_PATH=$PWD'/'
echo $DEMO_DATA_PATH
echo $DEMO_DATA_PATH/standoff2conll/annotations/
#
#rm -rf $DEMO_DATA_PATH/csv
## Convert the standoff files to conll
#python2 $DEMO_DATA_PATH/standoff2conll/standoff2conll.py $DEMO_DATA_PATH/standoff2conll/annotations/hyperion
##
#
#rm -rf $DEMO_DATA_PATH/train
#rm -rf $DEMO_DATA_PATH/val
#rm -rf $DEMO_DATA_PATH/test
#
#
#mkdir -p $DEMO_DATA_PATH/train/
#mkdir -p $DEMO_DATA_PATH/val/
#mkdir -p $DEMO_DATA_PATH/test/
#
##Split files into train test val
#python $DEMO_DATA_PATH/standoff2conll/split2train.py $DEMO_DATA_PATH/csv $DEMO_DATA_PATH

cd ../../../
#
python vitaflow/bin/vf_engine.py --mode=train --config=vitaflow/playground/clientx/config.gin
#
#python vitaflow/app/main.py


#### python vitaflow/bin/vf_engine.py --mode=predict --config=vitaflow/playground/clientx/config.gin