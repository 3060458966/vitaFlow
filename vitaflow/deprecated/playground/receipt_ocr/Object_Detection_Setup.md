
## Setup
> Please run object_detection_setup.sh for running all the following steps. Incase you have any issue, refer to this document.

**Prerequisites**
We recommend you to use python2 as the preferred interpretor.  Reason being the Tensorflow Object Dectection API, on top of which this full experiment revolves, is written in python2. For all the experiments we have used **GEFORCE GTX 1080 Ti 11GB** without which the training time is extended by factor of 8 (3 hours on GPU = 24 hours in CPU)

This readme describes every step required to get going with your own object detection classifier:

## Steps
### 1. Folder Structure
Create a folder in your local system and name it **ROD** or anything you like. 
This will be our landing directory and will contain the full TensorFlow Object Detection framework, as well as training images, training data, trained classifier, configuration files and everything else needed for the object detection classifier. Clone the repository into this **ROD** folder. If done move next


### 2. Set up TensorFlow Directory and Anaconda Virtual Environment
Get started by creating a tmux session in the machine having GPU hardware (_you can skip tmux step if your system has graphics card or you plan to train locally_). The system must have anaconda installed. Please refer [website](https://conda.io/docs/user-guide/install/index.html)  for installing anaconda if not installed.
```bash 
ssh rpx@172.17.0.5
# One time setup
tmux new -s text_detection
conda create -n tensorflow-gpu python=2.7 pip 
source activate tensorflow-gpu
python --version
```



#### 2a. Install Tensorflow-GPU (skip this step if Tensorflow-GPU)
```bash
pip install --ignore-installed --upgrade tensorflow-gpu
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python

```
At the time of writing this post, tensorflow-gpu 1.9 was used as the target version.


#### 2b. Download TensorFlow Object Detection API repository from GitHub
```bash
cd {path_to_folder_containing_ROD}/ROD

# Perform git clone 
git clone https://github.com/tensorflow/models
```
The [object detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) itself also has [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). Please follow the full installation guide especially __COCO API installation__ and __Protobuf Compilation__ steps in order to prevent any errors. 
Do not forget to execute the steps shown below 
>Then you must compile the Protobuf libraries:

```bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

>Add `models` and `models/slim` to your `PYTHONPATH`:

```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

>_**Note:** This must be ran every time you open terminal, or added to your `~/.bashrc` file._



*Please perform the Testing the Installation step at the end to confirm Tensorflow Object Detection  API step*
```bash
python object_detection/builders/model_builder_test.py
```
Perform git clone and copy all the files from the repository to object_detection folder
```
git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
```
#### 2c. Download the MASK-RCNN-Inception-V2 model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy.  We re-trained the detector on the Faster-RCNN-Inception-V2 model, and the detection worked considerably better, but with a noticeably slower speed.

You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the RCNN models.

We have used the MASK-RCNN-Inception-V2 model. 
[Download the model here.](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) Open the downloaded mask_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the mask_rcnn_inception_v2_coco_2018_01_28 folder to the ~/ROD/models/research/object_detection folder.

#### 2d. Moving and placing  files in correct folder
 Move all the files present in this repository directly into the ~/ROD/models/research/object_detection directory.
 The folder should have these files and folders listed below:

 - doc/ _Folder for keeping the images_

 - mask_rcnn_inception_v2_coco_2018_01_28/ _Folder with Pre-trained model_ 
 - images/
	 - test/ _folder containing test images and corresponding annotations in xml_. 
	 - train/ _folder containing train images and corresponding annotations in xml_.
	 - test_labels.csv _intermediate form required for dataset generation_
	 - train_labels.csv _intermediate form required for dataset generation_
 - inference_graph/ _folder will hold compact form of the model (frozen graph) after completion of model training_
 - training/ _folder will detain the new model generated after model training for our task_
 - generate_tfrecord.py _py file for converting csv to tfrecord_
 - Object_detection_image.py _py file for detecting objects in static image_
 - Object_detection_video.py _py file for detecting objects in video_
 - Object_detection_webcam.py _py file for detecting objects using webcam_
 - resizer.py _bulk image resizer_  
 - xml_to_csv.py _py file for converting xml annotations into csv_
 - test.jpeg _file for testing model_
 - test1.jpeg _another file for testing model_


To train your own object detector, delete the following files (do not delete the folders):
- All files in /object_detection/images/train and /object_detection/images/test
- The “test_labels.csv” and “train_labels.csv” files in /object_detection/images
- All files in /object_detection/training
-	All files in /object_detection/inference_graph


### 3. Gather and Label Pictures
#### 3a. Gather Pictures
A deep learning model needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have variety of backgrounds, lighting conditions and should be of different size and should vary in image quality along with desired objects. There could be some images where the desired objects are obscured, overlapped with something else, or only halfway in the picture.


#### 3b. Label Pictures
The TF Object Detection API work with images with their annotation in [**Pascal VOC format**](http://host.robots.ox.ac.uk/pascal/VOC/). For this reason we used [Image annotation tool](https://github.com/frederictost/images_annotation_programme) that supports this feature for labelling the images.
![](./screenshots/image_annotation.png)
The tools saves a .xml file containing the label data for each image. The xml files are used to generate the TFRecords, which are one of the inputs to the TF trainer. Once the image are labelled , there should be one .xml file for each image in the /test and /train directories.

```xml
<annotation>
	<folder>collection_01/part_1</folder>
	<filename>11cityroom-rcpt-articleInline-v2.jpg</filename>
	<path/>
	<source>
		<database>Unknown</database>
	</source>
	<size_part>
		<width>190</width>
		<height>373</height>
		<depth>3</depth>
	</size_part>
	<segmented>0</segmented>
	<object>
		<name>Merchant</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>7.8571395874023</xmin>
			<ymin>12</ymin>
			<xmax>172.8571395874</xmax>
			<ymax>45</ymax>
		</bndbox>
	</object>
	<object>
		<name>Line items and their value</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>2.8571395874023</xmin>
			<ymin>242</ymin>
			<xmax>178.8571395874</xmax>
			<ymax>255</ymax>
		</bndbox>
	</object>
	<object>
		<name>Line items and their value</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>0</xmin>
			<ymin>254</ymin>
			<xmax>178.8571395874</xmax>
			<ymax>269</ymax>
		</bndbox>
	</object>
	<object>
		<name>Tax</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>2.8571395874023</xmin>
			<ymin>291</ymin>
			<xmax>184.8571395874</xmax>
			<ymax>303</ymax>
		</bndbox>
	</object>
	<object>
		<name>Total</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>7.8571395874023</xmin>
			<ymin>304</ymin>
			<xmax>179.8571395874</xmax>
			<ymax>318</ymax>
		</bndbox>
	</object>
</annotation>
```

  
