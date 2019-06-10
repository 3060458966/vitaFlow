#!/usr/bin/env bash

# Data path
# This setup uses the clientx folder for saving all the content.
# one can also use /opt/data/vitaflow_demo for the same.
# export DEMO_DATA_PATH=/opt/data/vitaflow_demo

export DEMO_DATA_PATH=$PWD'/'
echo $DEMO_DATA_PATH
echo $DEMO_DATA_PATH/standoff2conll/annotations/

# remove the existing files
rm -rf $DEMO_DATA_PATH/csv

# get the brat annotated files-standoff files
# place it in the annotation folder in standoff2conll
#python2 $DEMO_DATA_PATH/standoff2conll/standoff2conll.py $DEMO_DATA_PATH/standoff2conll/annotations/{foldername}
#

# Convert the standoff files to conll
python2 $DEMO_DATA_PATH/standoff2conll/standoff2conll.py $DEMO_DATA_PATH/standoff2conll/annotations/hypera
##

# Recreate the train test val folders
rm -rf $DEMO_DATA_PATH/train
rm -rf $DEMO_DATA_PATH/val
rm -rf $DEMO_DATA_PATH/test
#

#
mkdir -p $DEMO_DATA_PATH/train/
mkdir -p $DEMO_DATA_PATH/val/
mkdir -p $DEMO_DATA_PATH/test/
#

#Split files into train test val
python $DEMO_DATA_PATH/standoff2conll/split2train.py $DEMO_DATA_PATH/csv $DEMO_DATA_PATH

cd ../../../

# DL training code
python vitaflow/bin/run_experiments.py --mode=train --config=vitaflow/playground/clientx/config.gin

# DL prediction code
python vitaflow/bin/run_experiments.py --mode=predict --config=vitaflow/playground/clientx/config.gin
