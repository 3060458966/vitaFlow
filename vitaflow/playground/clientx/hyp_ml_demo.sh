#!/usr/bin/env bash

# Data path

export DEMO_DATA_PATH=$PWD'/'
echo $DEMO_DATA_PATH
echo $DEMO_DATA_PATH/standoff2conll/annotations/

# remove the existing files
rm -rf $DEMO_DATA_PATH/csv

# get the brat annotated files-standoff files
# place it in the annotation folder in standoff2conll
#python2 $DEMO_DATA_PATH/standoff2conll/standoff2conll.py $DEMO_DATA_PATH/standoff2conll/annotations/{foldername}


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

# Split files into train test val
python $DEMO_DATA_PATH/standoff2conll/split2train.py $DEMO_DATA_PATH/csv $DEMO_DATA_PATH

cd ../../../

# Start the notebook for training
jupyter notebook CRF_NER-Iteration2.ipynb
