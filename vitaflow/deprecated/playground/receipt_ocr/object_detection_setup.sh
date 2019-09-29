#! /bin/bash

## GPU - ENV Setup
echo "Setting up GPU Virtual Env for Object Detection"
sleep 0.5
conda create -n tensorflow-receipts python=2.7 pip

mkdir ROD
cd ROD
MAINFOLDER=${PWD}

## Local ENV
echo "Installing Dependencies required for Object Detection"
sleep 0.5
source activate tensorflow-gpu

pip install --ignore-installed --upgrade tensorflow-gpu
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install jupyter
pip install matplotlib
pip install pandas contextlib2
pip install opencv-python

cd $MAINFOLDER
echo "Downloading - Tensorflow Model Framework"
sleep 0.5
git clone https://github.com/tensorflow/models

cd $MAINFOLDER
echo "Downloading - Object Detection Utils"
sleep 0.5
git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
mv TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/* models/research/object_detection/

cd $MAINFOLDER
echo "Downloading - Object Detection Pre-trained Model"
sleep 0.5
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
mv mask_rcnn_inception_v2_coco_2018_01_28/* models/research/object_detection/
## removing examples of pre-trained model
rm -rf models/research/object_detection/images/*

cd $MAINFOLDER
echo "Downloading - cocoapi - open image pre-trained model(tensorflow requirement)"
sleep 0.5
git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI
make
cp -r pycocotools models/research/

brew install protobuff
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


echo "Testing the Model"
sleep 0.5
cd $MAINFOLDER/models/research
python object_detection/builders/model_builder_test.py


echo 'Summary: tensorflow-receipts'
echo 'Create Conda Env: tensorflow-receipts (cmd: source activate tensorflow-receipts)'
echo 'Train Images Loc: models/research/object_detection/images/train/'
echo 'Test Images Loc: models/research/object_detection/images/train/'
