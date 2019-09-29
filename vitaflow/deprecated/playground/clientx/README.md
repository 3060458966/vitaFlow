
# VitaFlow - VideoImageTextAudioFlow
 ![](../../vitaflow-icon.png)


## Introduction

In this experiment we take up the task of solving NER on a real-world dataset . 
We will be creating a new experiment from scratch which would provide a better insight in dealing with new datasets. 

## Problem Statement

The objective is to build a model with a dataset that can detect and tag different types of named entities in a given document.



## Setup

### Getting Started

In order to setup and install dependencies kindly refer the [this](examples/conll2003/README.md),

### Folder Structure

The focus on folder named examples. Perform the following steps.

1. Create  a new folder called **clientx** 
2. Create a config.py. This file can be a copy of **conll2003/config.py** 
3. Change the necessary values like entity column and word column. 
4. Change the experiment and dataset name to match out *clientx* affix

```
~\vitaFlow
    ├──\examples
        ├──\clientx
            ├── clientx_dataset.py
            ├── config.py
            ├── __init__.py

```

The clientx folder houses the configurations for the whole experiment which is passed to the running engine.
The config.py contains the location of  dataset directories, parameters and hyper-parameters. clientx_dataset.py will handle all the dataset related operations.

```reStructuredText
import os

**experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow-clientx/")**
**experiment_name = "clientx_dataset"**
use_char_embd = True

experiments = {
    "num_epochs": 6,
    "dataset_class_with_path": **"examples.clientx.clientx_dataset.CLIENTXDataset"**,
    "iterator_class_with_path": "vitaflow.data.text.iterators.CSVSeqToSeqIterator",
    "model_class_with_path": "vitaflow.models.text.seq2seq.BiLSTMCrf",
    "save_checkpoints_steps" : 50,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : 25,
    "log_step_count_steps" : 10,

    # dataset - details
    **"examples.clientx.clientx_dataset.CLIENTXDataset"**: {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "minimum_num_words": 5,
        "over_write": False,
    },

    # data iterator
    "vitaflow.data.text.iterators.CSVSeqToSeqIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": **"clientx_data_iterator"**,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "text_col": **"word"**,
        "entity_col": **"entity_name_iob"**,
        "batch_size": 16,
        "seperator": "~",  # potential error point depending on the dataset
        "quotechar": "^",
        "max_word_length": 20,
        "use_char_embd": use_char_embd
    },

    # data model
    "vitaflow.models.text.seq2seq.BiLSTMCrf": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "use_char_embd": use_char_embd,
        "learning_rate": 0.001,
        "word_level_lstm_hidden_size": 128,
        "char_level_lstm_hidden_size": 300,
        "word_emd_size": 128,
        "char_emd_size": 300,
        "num_lstm_layers": 2,
        "keep_propability": 0.4,
    }
}
```



You can read more on this at [architecture]()

## About Dataset 

- The clientx dataset is a tagged at word level. 

- ```reStructuredText
  doc_id,word,x_cord,y_cord,pg_number,entity_name
  2xxx8,UxxxD,135.06,621.58,1.0,O
  2xxx8,SxxxS,135.06,621.58,1.0,O
  2xxx8,PxxT,135.06,621.58,1.0,O
  2xxx8,AxD,135.06,621.58,1.0,O
  2xxx8,TxxxxK,135.06,621.58,1.0,O
  2xxx8,OxxxE,135.06,621.58,1.0,O
  2xxx8,BxxxE,141.481,541.061,1.0,O
  2xxx8,TxxE,141.481,541.061,1.0,O
  2xxx8,PxxxT,141.481,541.061,1.0,O
  2xxx8,TxxxL,141.481,541.061,1.0,O
  2xxx8,AxD,141.481,541.061,1.0,O
  2xxx8,AxL,141.481,541.061,1.0,O
  2xxx8,BxxxD,141.481,541.061,1.0,O
  2xxx8,AxxxX,256.56,444.461,1.0,party_name__1
  2xxx8,CxxP.,256.56,444.461,1.0,party_name__1
  2xxx8,AxxX,256.56,444.461,1.0,party_name__1
  2xxx8,",",256.56,444.461,1.0,party_name__1
  2xxx8,IxC.,256.56,444.461,1.0,party_name__1
  2xxx8,Pxxxxxr,278.76,428.381,1.0,O
  ```

- Before we start feeding the data into training pipeline, we would like to convert it into [IOB](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) notation, so that we can mark the start and the end of the tagging sequence.  Here, the first word of the entity grouping is prefixed with **B-** and remain words in sequence are prefixed with **I-**.

- Words to ignore are tagged with O 

- ```reStructuredText
  doc_id,word,x_cord,y_cord,pg_number,entity_name,flag,entity_name_iob
  2xxx8,UxxxD,135.06,621.58,1.0,O,O
  2xxx8,SxxxS,135.06,621.58,1.0,O,O
  2xxx8,PxxT,135.06,621.58,1.0,O,O
  2xxx8,AxD,135.06,621.58,1.0,O,O
  2xxx8,TxxxxK,135.06,621.58,1.0,O,O
  2xxx8,OxxxE,135.06,621.58,1.0,O,O
  2xxx8,BxxxE,141.481,541.061,1.0,O,O
  2xxx8,TxxE,141.481,541.061,1.0,O,O
  2xxx8,PxxxT,141.481,541.061,1.0,O,O
  2xxx8,TxxxL,141.481,541.061,1.0,O,O
  2xxx8,AxD,141.481,541.061,1.0,O,O
  2xxx8,AxL,141.481,541.061,1.0,O,O
  2xxx8,BxxxD,141.481,541.061,1.0,O,O
  2xxx8,AxxxX,256.56,444.461,1.0,party_name__1,B-party_name__1
  2xxx8,CxxP.,256.56,444.461,1.0,party_name__1,I-party_name__1
  2xxx8,AxxX,256.56,444.461,1.0,party_name__1,I-party_name__1
  2xxx8,",",256.56,444.461,1.0,party_name__1,I-party_name__1
  2xxx8,IxC.,256.56,444.461,1.0,party_name__1,I-party_name__1
  2xxx8,Pxxxxxr,278.76,428.381,1.0,O,O
  ```

- This created dataset is then placed in the folder **vita-temp**

- ```reStructuredText
  ~/Vita-temp
      ├── **train**
      ├── **val**
      ├── **test**
  ```

- In the file examples/clientx/clientx_dataset.py, update the temp-data location to point  vita-temp

- ```reStructuredText
  ~\vitaFlow
      ├──\examples
          ├──\clientx
              ├── **clientx_dataset.py**
  ```


```reStructuredText
hparams.update({
            "experiment_name": "CLIENTXDataset",
            "minimum_num_words": 5,
            "over_write": False,
            **"temp-data": os.path.join(os.path.expanduser("~"), "vita-temp/"),**
        })	
```

## Experimentation

### 1. Train

```bash
python vitaflow/bin/run_experiments.py \
	--mode=train \
	--config_python_file=examples/clientx/config.py 
```

### 2. Retrain

```bash
python vitaflow/bin/run_experiments.py \
	--mode=retrain \
	--config_python_file=examples/clientx/config.py 
```

### 3. Predict

<!-- TODO: Tree -->

Place all the files to be predicted in the val folder 

1. ```reStructuredText
   clientx_dataset
       ├── BiLSTMCrf    
       ├── clientx_data_iterator
       ├── preprocessed_data
       │   ├── test
       │   ├── train
       │   └── **val**
   ```

2. ```bash
   python vitaflow/bin/run_experiments.py \
   	--mode=predict \
   	--config_python_file=examples/clientx/config.py 	
   ```

