
## 4. Generating Training Data

For training your model, please copy the train-test images along with their annotation XML files to '~/ROD/models/research/object_detection/images' folder.

### 1) Create the TensorFlow Records

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the /object_detection folder, issue the following command in the Anaconda command prompt:
```
(tensorflow-gpu) ~/ROD/models/research/object_detection python xml_to_csv.py
```
This creates a train_labels.csv and test_labels.csv file in the /object_detection/images folder. 

Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file in Step 5b. 

For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in `generate_tfrecord.py`:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    .....
```
With this:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == "merchant":
        return 1
    elif row_label == "receipt_number":
        return 2
    elif row_label == "date" :   
        return 3
    elif row_label == "line_items_and_value":
        return 4
    elif row_label == "total":
        return 5
    elif row_label == "tax":
        return 6
    elif row_label == "mode_of_payment":
        return 7
    else:
        None
```
Then, generate the TFRecord files by issuing these commands from the /object_detection folder:

```
python object_detection/generate_tfrecord.py --csv_input=object_detection/images/train_labels.csv --image_dir=object_detection/images/train --output_path=object_detection/train.record

python object_detection/generate_tfrecord.py --csv_input=object_detection/images/test_labels.csv --image_dir=object_detection/images/test --output_path=object_detection/test.record
```
These generate a train.record and a test.record file in /object_detection. These will be used to train the new object detection classifier.

### 5. Create Label Map and Configure Training
The last thing to do before training is to create a label map and edit the training configuration file.

#### 5a. Label map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the ~/ROD/models/research/object_detection/training folder. 
(Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for our Text Detector):
```
item {
 id: 1
 name: 'merchant'
}

item {
 id: 2
 name: 'receipt_number'
}

item {
 id: 3
 name: 'date'
}

item {
 id: 4
 name: 'line_items_and_value'
}

item {
 id: 5
 name: 'total'
}

item {
 id: 6
 name: 'tax'
}

item {
 id: 7
 name: 'mode_of_payment'
}

  

  

```
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. 


#### 5b. Configure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to ROD/models/research/object_detection/samples/configs and copy the faster_rcnn_inception_v2_pets.config file into the /object_detection/training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .
- Line 110. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "~/ROD/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "~/ROD/models/research/object_detection/train.record"
  - label_map_path: "~/ROD/models/research/object_detection/training/labelmap.pbtxt"

- Line 132. Change num_examples to the number of images you have in the /images/test directory.

- Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "~/ROD/models/research/object_detection/test.record"
  - label_map_path: "~/ROD/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 6. Run the Training
Here we go! From the /object_detection directory, issue the following command to begin training:
```
CUDA_VISIBLE_DEVICES=0 python object_detection/model_main.py --logtostderr --model_dir=object_detection/training/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28_regularized_clipping --pipeline_config_path=object_detection/training/pipeline.config

```
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins. When inference begins, after training, it will look like this:



Each step of training reports the loss. It will start high and get lower and lower as training progresses. For our training on the faster_rcnn_inception_v2_coco_2018_01_28 model, it started at about 3.0 and quickly dropped below 0.3. We recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.

You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the *text_detection* virtual environment, change to the ROD/models/research/object_detection directory, and issue the following command:
```
ROD/models/research/object_detection>tensorboard --logdir=training
```
This will create a webpage on your local machine at YourPCName:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.



The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. We typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

### 7. Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the /object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path object_detection/training/pipeline.config --trained_checkpoint_prefix object_detection/training/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28_regularized_clipping/model.ckpt-XXXX --output_directory object_detection/inference_graph
```
This creates a frozen_inference_graph.pb file in the /object_detection/inference_graph folder. The .pb file contains the object detection classifier.

### 8. Use Your Newly Trained Object Detection Classifier!
The object detection classifier is all ready to go! use the written Python scripts to test it out on an image.

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. ( currently NUM_CLASSES = 7.)

It will run your object detection model found at `output_inference_graph/frozen_inference_graph.pb` on all the images in the `test_images` directory and output the results in the `output/test_images` directory.













