#!/bin/bash
export DEMO_DATA_PATH=~/vitaflow_demo/
#1 input -> folder (hardcoded)
inputdirectory=$DEMO_DATA_PATH
#2. py2 -> stanoff2conll
#source activate quantipy2.7
#cd ./examples/clientx/standoff2conll

# clear annotations folder
rm -rf ./examples/clientx/standoff2conll/annotations
mkdir ./examples/clientx/standoff2conll/annotations
echo $1
# move the uploaded file into annotations folder {TODO}
cp $1 ./examples/clientx/standoff2conll/annotations/

# clear csv folder
rm -rf csv
mkdir csv
rm -rf postprocessed
mkdir -p postprocessed

sourcefile=`basename $1`
touch ./examples/clientx/standoff2conll/annotations/"${sourcefile%.*}.ann"
python2.7 ./examples/clientx/standoff2conll/standoff2conll.py ./examples/clientx/standoff2conll/annotations

# csv file in one place

# move files from csv folder to test folder {check}
# move files from csv folder to test folder {check}
#change here
rm -rf $inputdirectory'clientx_dataset/clientx_data_iterator/postprocessed/'
rm -rf $inputdirectory'clientx_dataset/preprocessed_data/test/'
rm -rf $inputdirectory'clientx_dataset/test_padded_data_True.p'


#change here
mkdir -p $inputdirectory'clientx_dataset/clientx_data_iterator/postprocessed/'
mkdir -p $inputdirectory'clientx_dataset/preprocessed_data/test/'


#change here
mv csv/* $inputdirectory'clientx_dataset/preprocessed_data/test/'

# run predictions
#source activate qPy3

CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/vf_engine.py --mode=predict --config_python_file=examples/clientx/config.py

#change here
mv $inputdirectory"clientx_dataset/clientx_data_iterator/postprocessed/" .
