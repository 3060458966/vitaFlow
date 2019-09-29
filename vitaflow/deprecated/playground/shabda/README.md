# Shabda

A DL based Audio enchancement solution


### [DeWave](https://github.com/chaodengusc/DeWave)

We have considered DeWave as a starting point to our exploration in Audio and/or Video sepration task.

DeWave is simplest of all which uses Deep Clustering to seperate two mixture of audio signals.

We have provided a [download.sh](download.sh) script that downloads data of size ~20GB and unzips the data to default location.

Please refer below on how to run the experiments.

DeWave repo is sucked into three python classes that resides as [tedlium_dataset.py](deprecated/tedlium_dataset.py), 
[tedlium_iterator.py](deprecated/tedlium_iterator_basic.py) and [deep_clustering.py](deep_clustering.py) respectively.


### Experiment

```
 find . -name "*.pyc" -exec rm -f {} \;
 sudo apt-get install ffmpeg
 sudo apt-get install spacy
```
**Cache the Preprocessed Data Manually**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py --mode=run_iterator -config_python_file=vitaflow/playground/shabda/config.py
```
**Train**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py --mode=train -config_python_file=vitaflow/playground/shabda/config.py
```
**Predict on single file**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py \
--mode=predict_instance \
--test_file_path=mixed.wav \
-config_python_file=vitaflow/playground/shabda/config.py
```


### Reference: 

- https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html
- https://github.com/zhr1201/Multi-channel-speech-extraction-using-DNN/blob/master/Manuscript-InterNoise2017-ZhouHaoran_0525.pdf
- https://towardsdatascience.com/audio-processing-in-tensorflow-208f1a4103aa

Look to Listen/Cocktail Party based on Tensorflow:   
**Deep Clustering**  
    - [https://github.com/chaodengusc/DeWave](https://github.com/chaodengusc/DeWave)  
    - [https://github.com/TotallyFine/deep-clustering](https://github.com/TotallyFine/deep-clustering)  
    
**Audio Sepration**  
    - [https://github.com/andabi/music-source-separation](https://github.com/andabi/music-source-separation)  
    - [https://github.com/eesungkim/Speech_Enhancement_DNN_NMF](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)  
    - [https://github.com/f90/AdversarialAudioSeparation](https://github.com/f90/AdversarialAudioSeparation)    
    - [https://github.com/philipperemy/deep-speaker](https://github.com/philipperemy/deep-speaker)
**Audio- Video**  
    - [https://github.com/avivga/audio-visual-speech-enhancement](https://github.com/avivga/audio-visual-speech-enhancement) (Keras)   
    - [https://github.com/andrewowens/multisensory](https://github.com/andrewowens/multisensory)  
    - [https://github.com/crystal-method/Looking-to-Listen](https://github.com/crystal-method/Looking-to-Listen)  
    - [https://github.com/Veleslavia/vimss](https://github.com/Veleslavia/vimss)  
    - [https://github.com/bill9800/speech_separation](https://github.com/bill9800/speech_separation)  

- https://github.com/vishwajeet97/Cocktail-Party-Problem
- https://github.com/marl/audiosetdl 


# Code Reference
- [DeWave](https://github.com/chaodengusc/DeWave)

## Paper
- [https://arxiv.org/abs/1508.04306](https://arxiv.org/abs/1508.04306)
- [https://arxiv.org/pdf/1705.04662.pdf](https://arxiv.org/pdf/1705.04662.pdf)

### Dataset:
- [TEDLium](https://projets-lium.univ-lemans.fr/ted-lium/)
- [TPS](http://www-mmsp.ece.mcgill.ca/Documents/Data/)
- [Audioset](https://research.google.com/audioset/)
- [Audioset Downloader](https://github.com/marl/audiosetdl)
### Audio Basics
- [Decibel](https://www.rapidtables.com/electric/decibel.html)
- http://www.cs.princeton.edu/~fiebrink/314/2009/week12/FFT_handout.pdf

### Maths Refresher:

**Imaginary Numbers**
    - https://www.youtube.com/watch?v=T647CGsuOVU   
    - [imaginary_numbers_are_real_rev2_for_screen.pdf](https://static1.squarespace.com/static/54b90461e4b0ad6fb5e05581/t/5a6e7bd341920260ccd693cf/1517190204747/imaginary_numbers_are_real_rev2_for_screen.pdf)
    - https://www.electronics-tutorials.ws/accircuits/complex-numbers.html  
    - Polar form vs Rectangular form  
**Euler's Formula**   
    - https://www.youtube.com/watch?v=m2MIpDrF7Es&t=23s  
    - https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/  
**FFT**  
    - https://www.youtube.com/watch?v=spUNpyF58BY   
    - https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/   

