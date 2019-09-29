# VitaFlow - VideoImageTextAudioFlow

 ![](../../vitaflow-icon.png)

<!-- TODO: please include spaces after all headings -->
<!-- TODO: please include a short summary -->
<!-- TODO: please include step to quickly create and run a sample study -->
<!-- TODO: please include (BEST USE CASES - NEW SECTION) sample cases where this tools can be used-->
<!-- TODO: Two level requirement.txt files, one for global and another is use case specific -->

## Introduction

In this example, we take up the [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) (Named Entity Recognition) problem to be solved on CONLL 2003 dataset. By end of this readme, you will be able to use vitaFlow for training a [seq2seq](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwi46dvqzqvfAhUVA3IKHQjyALgQFjAAegQICBAB&url=https%3A%2F%2Fgoogle.github.io%2Fseq2seq%2F&usg=AOvVaw0u7lKNTuRYZChT9y_lBmVW) model from scratch and run prediction on it. 

![Seq2Seq NER](https://www.depends-on-the-definition.com/wp-content/uploads/2017/10/many_to_many-945x489.png)

 



## Problem Statement

The objective is to build a model that can detect and tag different types of named entities in a given document.

```reStructuredText
-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER
```



## Dataset

The dataset is taken from language-independent named [entity recognition task](https://www.clips.uantwerpen.be/conll2003/ner/). This dataset has two subsets English and German but we use English for the sense of brewity. 

## Setup

### Getting Started

Clone the repository 

```
git clone https://github.com/Imaginea/vitaFlow
cd vitaFlow
```

### Folder Structure

The folder to be focussed in this experiment is placed in examples. This folder houses various examples.

Consider conll2003 

<!-- TODO: please use unix `tree` tool -->

```
vitaFlow
├── examples
    ├── conll2003
    │   ├── config.py
    │   ├── config.yml
    │   ├── conll_2003_dataset.py
    │   ├── __init__.py
    │   ├── __pycache__
    │   └── README.md
	├── __init__.py

```

This folder houses the configurations for the whole experiment which is passed to the running engine.
The config.py contains the location of  dataset directories, parameters and hyper-parameters. 
You can read more on this at [architecture]()

### Requirement

Python interpreter : Python 3  

Dependency and environment manager : Anaconda

```bash 
# One time setup
conda create -n vitaflow-gpu python3 pip 
source activate vitaflow-gpu
python --version
```

__Install the prerequisites__

```bash
pip install -r requirements.txt
```

Incase you don't have a GPU please install tensorflow CPU version

```
conda install tensorflow
```

## How to run

Like every machine learning project, we have train and predict mode which needs to be passed as an argument while running the experiment.
The entry point in the system is *run_experiments.py*. The arguments passed to it are the config file and mode in which it has to run. By default it supports three modes (train, retrain and predict).

### 1. Train

```bash
python vitaflow/bin/run_experiments.py \
	--mode=train \
	--config=vitaflow/playground/conll2003/config.gin
```

We start the training by passing **--mode = train** . *run_experiments.py* will pickup the config file provided to it  and look for the dataset. 

**--option= train** is an optional parameter and defaults to train when not present.

If the dataset is not present on the drive, the **conll_2003_dataset.py** will download (place it in raw_data), preprocess (keep it preprocessed_data) and keep the data ready. There will be 3 different folders train, val and test.
<!-- TODO: please use unix `tree` tool -->

```reStructuredText
conll_2003_dataset
    ├── BiLSTMCrf    
    ├── conll_data_iterator
    ├── preprocessed_data
    │   ├── **test**
    │   ├── **train**
    │   └── **val**
    ├── raw_data
    │   ├── conll2003.zip
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt

```

### 2. Retrain

Incase the experiments needs to be repeated for more number of epochs or requires to be resumed from earlier point of interruption, it can done so using the following command. It will continue the training from the last point.

```bash
python vitaflow/bin/run_experiments.py \
	--mode=retrain \
	--config=vitaflow/playground/conll2003/config.gin 
```

### 3. Predict

<!-- TODO: Tree -->

Place all the files to be predicted in the val folder 

1. ```reStructuredText
   conll_2003_dataset
       ├── BiLSTMCrf    
       ├── conll_data_iterator
       ├── preprocessed_data
       │   ├── test
       │   ├── train
       │   └── **val**
   ```

2. ```bash
   python vitaflow/bin/run_experiments.py \
   	--mode=predict \
   	--config_python_file=examples/conll2003/config.py 	
   ```

3. The result of the output will be generated in the predictions folder.

4. The predictions will be placed in the folder



<!-- TODO: Tree -->

```reStructuredText
$HOME
├──{experiment_root_directory}
    ├──{experiment_name}
       ├── {iterator_name}
          └── predictions
```



In current scenario, the location of the prediction folder would be the following.



```reStructuredText
$HOME
├──vitaFlow
    ├──conll_2003_dataset
        ├──conll_data_iterator
            └──predictions
​```
```



## Extra Notes

1.  The use of this example does not restrict the user to only solving named entity recognition.
2. It can be further extended to any seq2seq task like Machine Translation, Aspect Extraction for Sentiment Analysis etc. 

<!-- TODO: New file - contribution.md -->
<!-- TODO: New file - relases.md -->

<!-- TODO: New file - QuickStart.md - FUTURE -->
<!-- TODO: New file - FirstExample.md - FUTURE -->
