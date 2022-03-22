"""
File: plot_emo_embeddings.py

Authors:
    - Eric Zhou

Date:
    - Created: Nov 29, 2021

Purpose:
    - Plot the vector embeddings of speech files from
    the Emotion Speech Dataset (https://kunzhou9646.github.io/controllable-evc/)
    given a pre-trained model from SpeechBrain (https://speechbrain.github.io/),
    an open-source library for speech processing
    - This is a standalone script

Emotion Speech Dataset Source:
    - Kun Zhou, Berrak Sisman, Mingyang Zhang and Haizhou Li,
    "Converting Anyone's Emotion: Towards Speaker-Independent Emotional Voice Conversion",
    in Proc. INTERSPEECH, Shanghai, China, October 2020.
    - https://kunzhou9646.github.io/controllable-evc/

List of Pre-Trained Models from SpeechBrain:
    - https://huggingface.co/speechbrain

Setup (install speechbrain):
    - git clone https://github.com/speechbrain/speechbrain.git
    - cd speechbrain
    - pip install -r requirements.txt
    - pip install --editable .

Args:
    --hf_model_id <Hugging Face Model ID>
        - Type: String
        - Description: The speech encoder to evaluate
        - Default value: "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

Usage:
    plot_emo_embeddings.py --hf_model_id <optional: Hugging Face Model ID>
"""

import argparse
import os
import torch
import torch.utils.data as data_utils
import yaml

import stargan.my_dataset as my_dataset
from utils import audio_utils

from speechbrain.pretrained.interfaces import foreign_class


def main(hf_model_id: str) -> None:
    """ Plot audio embeddings on a 2D plane using PCA

    Args:
        hf_model_id: [String] Hugging Face model ID
        dataset_path: [String] Path to dataset, assumes dataset directory will have the same
            file structure as the Emotion Speech Dataset mentioned in file docstring

    Assumes:
        configs/config_step1.yaml exists and contains proper configurations for loading
        the MyDataset class

    Returns:
        Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = foreign_class(source=hf_model_id,
                               pymodule_file="custom_interface.py",
                               classname="CustomEncoderWav2vec2Classifier",
                               run_opts={"device": device})

    ###
    #   Create dataloader
    ###
    dataset_config = yaml.safe_load(open("configs/config_step1.yaml", 'r'))
    world_dir = os.path.join(dataset_config["data"]["dataset_dir"], "world")
    filenames = my_dataset.get_filenames(world_dir)

    dataset = my_dataset.MyDataset(dataset_config, filenames)

    data_loader = data_utils.DataLoader(dataset, batch_size=64,
                                        collate_fn=my_dataset.collate_length_order,
                                        num_workers=0, shuffle=False)

    ###
    #   Loop through audio files
    ###
    for _, (wav_forms, wav_lengths), emotion_labels in data_loader:
        encoded_batch = classifier.encode_batch(wav_forms, wav_lengths)
        print("Encoded batch", encoded_batch.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process args for plot_emo_embeddings.py.')

    parser.add_argument('--hf_model_id', type=str, default='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
                        help='Hugging Face model name, e.g., speechbrain/emotion-recognition-wav2vec2-IEMOCAP')

    args = parser.parse_args()
    main(args.hf_model_id)
