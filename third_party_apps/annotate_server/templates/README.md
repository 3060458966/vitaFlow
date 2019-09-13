# Images Annotation

### 1. Start Server
```
cd annotate_server

sudo apt install php7.0-cli

php -S localhost:8000
```

### 2. Configuration


To customize the directories used, edit the PHP file **inc/configuration.php**
```php
<?php
# Image path to be used in the HTML client
$IMAGE_WEB_DIR = "data/images";

# Image path for internal PHP use
$IMAGE_ROOT_DIR  = "/Users/sampathm/Desktop/test_images";
$ANNOTATIONS_DIR = "/Users/sampathm/Desktop/test_images2";

# Collection name 
$COLLECTION_NAME = "collection_01";

# Not annotated image 80% to be presented to user
$ratio_new_old = 80;
?>
```
### 3. Images
Images to be annotated are located in **data/images/collection_01/**

### 4. List of classes

The list of classes can be customized in the file **resources/list_of_tags.json**
```json
[
        {"name": "Merchant", "icon": ""},
        {"name": "Receipt number", "icon": ""},
        {"name": "Date", "icon": ""},   
        {"name": "Line items and their value", "icon": ""},
        {"name": "Total", "icon": ""},
        {"name": "Tax", "icon": ""},
        {"name":"Mode of payment","icon":""}
]

```

### 5. Annotations Target directory 
Each image will generate one XML file in the directory **data/annotations**

## Output as Pascal VOC xml files

This format is a standard and can be easily read from [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)

```xml
<?xml version="1.0"?>
<annotation>
  <folder>collection_01/part_1</folder>
  <filename>pexels-photo-60091.jpg</filename>
  <path/>
  <source>
    <database>Unknown</database>
  </source>
  <size_part>
     <width>1125</width>
     <height>750</height>
     <depth>3</depth>
  </size_part>
  <segmented>0</segmented>
  <object>
    <name>Bird</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>488</xmin>
      <ymin>245.5</ymin>
      <xmax>674</xmax>
      <ymax>601.5</ymax>
    </bndbox>
  </object>
</annotation>
```
