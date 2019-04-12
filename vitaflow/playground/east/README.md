# EAST: An Efficient and Accurate Scene Text Detector

**Refernce Git** : https://github.com/argman/EAST  
**Paper** : https://arxiv.org/abs/1704.03155v2   
**Dataset ICDAR** : [Text Localization 1: 2013](http://rrc.cvc.uab.es/?ch=2&com=downloads),[2015](http://rrc.cvc.uab.es/?ch=4&com=introduction), [2019](http://rrc.cvc.uab.es/?ch=13)  
**Imaginea Folks** :   

- Use Google Drive Link: https://drive.google.com/drive/folders/1CmEkiDHWQB-miGQRSU3pK7qoAfoifb-R?usp=sharing and download the  files

```sh
cd /opt/data/icdar
tree .
.
├── 2013
│   ├── task1
│   │   ├── Challenge2_Test_Task12_Images.zip
│   │   ├── Challenge2_Test_Task1_GT.zip
│   │   ├── Challenge2_Training_Task12_Images.zip
│   │   └── Challenge2_Training_Task1_GT.zip
│   └── task2
│       ├── Challenge2_Test_Task12_Images.zip
│       ├── Challenge2_Test_Task2_GT.zip
│       ├── Challenge2_Training_Task12_Images.zip
│       └── Challenge2_Training_Task2_GT.zip
├── 2015
│   ├── ch4_test_images.zip
│   ├── ch4_training_images.zip
│   ├── ch4_training_localization_transcription_gt.zip
│   └── Challenge4_Test_Task1_GT.zip
└── 2019
    ├── 0319updated.task1train(628p)-20190320T064205Z-001.zip
    ├── 0319updated.task2train(628p)-20190320T050248Z-001.zip
    └── task1_test(361p)-20190404T055401Z-001.zip
```

**2015** : We have train and test data
**2018** : We have only train data

## Text Localization:
- All images are provided as JPEG or PNG files and the text files are UTF-8 files with CR/LF new line endings.
- The ground truth is given as separate text files (one per image) where each line specifies the coordinates of one word's bounding box and its transcription in a comma separated format 
- [2015](http://rrc.cvc.uab.es/?ch=4&com=tasks)

img_1.txt <-> gt_img_01.txt

```sh
x1, y1, x2, y2, x3, y3, x4, y4, transcription
```

- [2019](http://rrc.cvc.uab.es/?ch=13&com=tasks)

img_1.txt <-> img_01.txt

```sh
x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2, transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3, transcript_3
```

## Data Setup

```sh
cd /opt/data/icdar/
mkdir train
mkdir test
mkdir val

#train files
#2013,2015
# unzip 2013/task2/Challenge2_Training_Task12_Images.zip -d 2013_train/
# unzip 2013/task2/Challenge2_Training_Task2_GT.zip -d 2013_train/
# unzip 2015/ch4_training_images.zip -d 2015_train/
# unzip 2015/ch4_training_localization_transcription_gt.zip -d 2015_train/
unzip 2019/0319updated.task1train\(628p\)-20190320T064205Z-001.zip -d 2019_train/
cd 2019_train/0319updated.task1train\(628p\)/
mv * ../
cd ..
rm -rf 0319updated.task1train\(628p\)/
cd ../

# cd 2013_train/
# rename 's/GT/gt/' *
# cd ..
# mv 2013_train/* train/
# mv 2015_train/* train/

cd 2019_train/
mv `ls  | head -50` ../test/
mv `ls  | head -100` ../val/
cd ..
mv 2019_train/* train/

# rm -rf 2013_train/
# rm -rf 2015_train/
rm -rf 2019_train/


# test and validation
#unzip 2015/ch4_test_images.zip -d 2015_test_images/
#unzip 2015/Challenge4_Test_Task1_GT.zip -d 2015_test_txt/

#move first 100 images for testing and rest for validation
#cd 2015_test_images/
#mv `ls  | head -100` ../test/
#mv *.* ../val/
#cd ..

#cd 2015_test_txt/
#mv `ls  | head -100` ../test/
#mv *.* ../val/
#cd ..

#rm -rf 2015_test_images/
#rm -rf 2015_test_txt/
ls
```

## Requirements

```sh
pip install tensorflow-gpu
pip install tensorflow-serving-api
pip install gin-config
pip install numpy
pip install opencv-python
pip install shapely
pip install tqdm
pip install matplotlib
pip install scipy
pip install grpcio
```

## Configuration

Check this [file](config.gin)!

##Compile lanms

```sh
cd lanms
make
```
If you get compilation issues with lanms, follow this [link](https://github.com/argman/EAST/issues/156#issuecomment-404166990)! to resolve it.

## Running

```sh
#train
python main.py

#predict
python main --predict=true

#serving
export MODEL_NAME=EAST
export MODEL_PATH=/opt/tmp/icdar/east/EASTModel/exported/

tensorflow_model_server   \
--port=8500   \
--rest_api_port=8501   \
--model_name="$MODEL_NAME" \
--model_base_path="$MODEL_PATH"

python grpc_predict.py \
  --image /opt/tmp/test/img_967.jpg \
  --output_dir /opt/tmp/icdar/ \
  --model EAST  \
  --host "localhost" \
  --signature_name serving_default


python grpc_predict.py \
  --images_dir /opt/tmp/test/ \
  --output_dir /opt/tmp/icdar/ \
  --model EAST  \
  --host "localhost" \
  --signature_name serving_default 
```

### PS

- As compared to original EAST repo, we have used Tensorflow high level APIs tf.data and tf.Estimators
- This comes in handy when we move to big dataset or if we wanted to experiment with different models/data
- TF Estimator also takes care of exporting the model for serving! [Reference](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421)
