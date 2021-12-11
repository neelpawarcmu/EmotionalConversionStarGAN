# Emotional Voice Conversion Using StarGANs
This repository is cloned from contains an extension to the baseline model from ICASSP 2020 paper "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition"
for our Introduction to Deep Learning project (11-785) at Carnegie Mellon University. 

The original repository can be found here: https://github.com/glam-imperial/EmotionalConversionStarGAN

We have applied a Speech Emotion Recognition model to this from SpeechBrain as part of our extension: https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP

The original paper used the IEMOCAP dataset: https://sail.usc.edu/iemocap/

We are instead using the Emotional Speech Dataset (ESD),
since it is newer and more comparable to recent baselines: https://kunzhou9646.github.io/controllable-evc/

# Running and Training Model 
**- Requirements:**
* python>3.7.0
* pytorch
* numpy
* argparse
* librosa
* scikit-learn
* tensorflow < 2.0 (just for logging)
* pyworld
* matplotlib
* yaml

If running Anaconda(eg. Deep Learning AMI in AWS), you will have to `conda activate pytorch_p37` and `pip install librosa pyworld tensorflow==1.15`.

**- Clone repository:**
```
git clone https://github.com/eric-zhizu/EmotionalConversionStarGAN.git
cd EmotionalConversionStarGAN
```

**- Repositories:**

We have three primary repositories with three version of our model: *master*, *ser-embed-model-1*, and 
*ser-embed-model-5*. 
More details can be found [here](#model-implementations)


**- Download ESD dataset from https://kunzhou9646.github.io/controllable-evc/**

Running the script **run_preprocessing.py** will prepare the data as needed for training the model. It assumes that ESD is already downloaded and is stored in an arbitrary directory <DIR> with this file structure
```
<DIR>
  |- Speaker 0001  
  |     |- Annotations (0001.txt) 
  |     |- Angry/
  |     |-    |- evaluation
  |     |-    |-    |- 0001_<recording id>.wav
  |     |-    |- test
  |     |-    |- train
  |     |- Happy/
  |     |- ...
  |- ...
  |- Speaker 0020
  |     |- Annotations (0020.txt) 
  |     |- Angry/
  |     |- Happy/
  |     |- ...
```
where Annotations is a text file holding the emotion labels for each recording
  
 To preprocess run
 ```
 python run_preprocessing.py --iemocap_dir <DIR> 
 ```
 which will move all audio files to ./procesed_data/audio as well as extract all WORLD features and labels needed for training.
 It will only extract these for samples of the correct emotions (angry, sad, happy) and under the certain hardocded length threshold (to speed up training time). it will also create dictionaries for F0 statistics which are used to alter the F0 of a sample when converting.
After running you should have a file structure:
```
./processed_data
 |- annotations
 |- audio
 |- f0
 |- labels
 |- world
 ```
 
`run_preprocessing.py` will take a few hours. We recommend saving `processed_data` somewhere. If you are in a rush and trying to test whether the model will work, you can interrupt the script after it has converted only a subset of the data.
 

# SpeechBrain (SER)

 To run code with the SER embeddings, SpeechBrian and its dependencies need to be installed. Speech brain can be installed by cloning the speechbrain repository:
```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```
Steps taken from and more details at:
https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP


In our project, we load the Emotion Recognition using the following code:
```angular2html
from speechbrain.pretrained.interfaces import foreign_class
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
print(text_lab)
```

This can be found in ```stargan/model.py``` in branches ```ser_embed_model_1``` and ```ser_embed_model_5```

 # Training EmotionStarGAN
 Main training script is **train_main.py**. However to automatically train a three emotion model (angry, sad, happy) as it was trained for "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition", simply call:
 ```
 ./full_training_script.sh
 ```
 This script runs three steps:
 1. Runs classifier_train.py - Pretrains an auxiliary emotional classifier. Saves best checkpoint to ./checkpoints/cls_checkpoint.ckpt
 2. Step 1: Runs main training for 200k iterations in --recon_only mode, meaning model learns to simply reconstruct the input audio.
 3. Step 2: Trains model for a further 100k steps (or what we configure in ```config_model_step_2```), introducing the pre-trained classifier.
 
 A full training run will take ~24 hours on a decent GPU. The auxiliary emotional classifier can also be trained independently using **classifier_train.py**.
 
 # Sample Conversion
 Once a model is trained you can convert the output audio samples using **convert.py**. Running
 ```
 python convert.py --checkpoint <path/to/model_checkpoint.ckpt> -o ./processed_data/converted
 ```
 will load a model checkpoint and convert all samples from the train set and test set into each emotion and save the converted samples in /processed_data/converted

# Model Implementations

We have three primary model implementation in three different repositories:

1. master: Baseline model with modifications to use the DeepEST dataset
2. StarGAN-Embed-1: Code in ser-embed-model-1. Model extension with SER continuous embedding.
   In this model, we concatenate with the encoded representation before one of the upsampling layers.
   We change our upsampling channel to have the emotion embedding size which is 768.
3. StarGAN-Embed-5: Code in ser-embed-model-5. Model extension with SER continuous embedding.
   In this model, we concatenate with the encoded representation before all the upsampling layers. We change our upsampling 
   channel to instead upsample with respect to the emotion embedding size of 768 (instead of the num_of_classes). 
   
# Code Structure 

The code is organized as follows (with most notable files files)

```
EmotionalConversionStarGAN
  |- stargan  (Contains most of the model specific code)
  |     |- classifiers.py
  |     |- model.py
  |     |- my_dataset.py
  |     |- solver.py
  |     |- ...
  |- ...
  |- utils (Contains most of the Audio processing and data processing utils)
  |     |- audio_utils.py
  |     |- data_preprocessing_utils.py
  |     |- preprocess_world.py
  |     |- ...
  |- configs (Configs like lr, step sizes, num of emotions) 
  |     |- config_step1.py
  |     |- config_step2.py
  |  mcd_evaluate.py
  |  convert.py
  |  full_training_script.sh
  |  classifier_train.py
  |  run_preprocessing.py
  |  train_main.py
  |- notebooks (Contains all the notebooks used to generate plots etc)
  |     
```

###stargan

The ```stargan``` folder has the core model architecture and the code required to run the model.

**-model.py** : Contains the main model architecture ```StarGAN_emo_VC1``` that included the generator and discriminator. 
```GeneratorWorld``` is the main generator and ```DiscriminatorWorld``` is the main discriminator used in this process. 
When applicable, we concatenated the continuous emotion embeddings here


**-solver.py** : Wrapper around the main model that instantiates the hyperparameters of the model

**-my_dataset.py** : Contains the main dataset ```My_Dataset``` and the dataloader for the model

**-classifiers.py**: Contains the emotion classifier


###utils

The ```utils``` folder had the files for preprocessing audio

**-audio_utils.py** : Utility files for audio

**-data_preprocessing_utils.py** : data preprocessing script 

**-preprocess_world.py** : Preprocessing file for WORLD features

###configs

Contains ```configs``` files for step1 and step2. 

###mcd_evaluate

Used to compute the Mel-ceptral distortion.

###convert.py

Runs a testing loop to convert the original files to the emotions given. 

###full_training_script.sh

Scripts to help kick off training. This can be used as a single point to start training.
More details [here](#training-emotionstargan)

###classifier_train.py 

Script to train the emotion classifier

###train_main.py

Trains the model for the steps passed in the config using solver.py

###notebooks
This folder contains notebooks we ran in Google Colab for various pre- and post-processing. 

**-mfcc_loss_calc.ipynb** : Calculate the MFCC loss

**-speechbrain_finetune.ipynb** : We finetuned the SER to increase its accuracy using this notebook

**-stargan_emo_embeddings.ipynb** : Loops through audio files, gets the emotion embeddings and saves them to disk

**-stargan_setup.ipynb** : Helper notebook to run training on Colab


# Contributions 

This project would not be possible without the baseline model from
"StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition", the DeepEST dataset, and SpeechBrain SER. The main contributors
of this project are Eric Zhou [@eric-zhizu], Stuti Misra [@stutimisra], 
Syeda Sheherbano Rizvi [@srizvi6], and Neel Pawar [@neelpawarcmu]. 