[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/imaginea/vitaflow/blob/master/LICENSE)


# VitaFlow - VideoImageTextAudioFlow
 ![](vitaflow-logo.png)

## [Introduction](what_is_vitaflow.md)

## Environment Setup

**Python Setup**

```
   git clone https://github.com/Imaginea/vitaFlow/
   cd vitaFlow/
   conda create -n vf python=3.5
   conda activate vf
   export SLUGIFY_USES_TEXT_UNIDECODE=yes
   pip install -r requirements.txt
```

**Ubuntu Specific Installation**
- https://www.tensorflow.org/tfx/serving/setup
- `sudo apt-get -y install postgresql postgresql-contrib libpq-dev postgresql-client postgresql-client-common`


---------------------------------------------------------------------------------------------------------

## Demo  
We have put up a small working demo, which basically can read text from camera images. (Note: the models are not tweaked for the maximum performance)   

The pipeline components includes:   
 - EAST Model
 - Image Binarization 
 - Cropping tool (crops each text line from the image as single image)
 - OCR with Tesseract and DL based OCR called Calamari  (image to text for all the images that were generated from previous cropper stage)
 - Text stiching (where each text out from the images are stiched as one text file)
 
### Local machine

**MakeFile**  

- Input Files Directory : [data/receipts/](data/receipts/)
- output  Files Directory : [data/text_out/](data/text_out/)
  - X51008142068_**CalamariOcrPlugin**.txt : **Output using our pipeline**
  - X51008142068_**TessaractOcrPlugin**.txt : **Ouput using PyTesseract**
  
```
python vitaflow/bin/vf-ocr.py --image_dir=data/receipts/ --out_dir=data/text_out/
```

**Web UI**
- TODO
