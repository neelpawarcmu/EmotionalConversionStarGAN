"""
Author - Max Elliott
Modifier - Eric Zhou ericzhou@andrew.cmu.edu

Script completes three task:
    1) refile the ESD dataset
    2) Generates the WORLD features needed for training EmotionalConversionStarGAN
    3) Generates f0 look up dictionaries needs for producing converted audio files
"""

import os
import numpy as np
import pickle
from shutil import copyfile
import argparse
from utils.data_preprocessing_utils import get_wav_and_labels, read_annotations
from utils.preprocess_world import world_features, cal_mcep, get_f0_stats

MIN_LENGTH = 0 # actual is 59
MAX_LENGTH = 1719

def copy_files(source_dir, output_dir):

    """
    Make initial directory structure needed for preprocessing. Takes source directory
    and puts all audio files in one folder, and all annotations in another.

    See the official repo for source_dir == IEMOCAP directory format
    Assumes source_dir has the same format as the data in this paper:
    - https://kunzhou9646.github.io/controllable-evc/

    Source directory:
    - speaker directory (e.g., 0001)
        - annotation.txt (e.g., 0001.txt)
            - tab-delimited: [wav file name without .wav extension] [speech text] [emotion label]
            - emotion labels are Angry, Happy, Neutral, Sad, Surprise
        - emotion directory (e.g., Angry)
            - train/test/validation directory
                - .wav files
    """
    audio_output_dir = os.path.join(output_dir, 'audio')
    annotations_output_dir = os.path.join(output_dir, 'annotations')

    if not os.path.exists(audio_output_dir):
        os.mkdir(audio_output_dir)
    if not os.path.exists(annotations_output_dir):
        os.mkdir(annotations_output_dir)

    for speaker in os.listdir(source_dir):
        speaker_dir = os.path.join(source_dir, speaker)

        if not os.path.isdir(speaker_dir):
            continue

        annotation_file = os.path.join(speaker_dir, speaker + ".txt")
        dest_file = os.path.join(annotations_output_dir, speaker + ".txt")
        # Copy annotations to dest annotations folder
        if not os.path.exists(dest_file):
            copyfile(annotation_file, dest_file)

        for root, _, files in os.walk(speaker_dir):
            for file in files:
                if file.endswith(".wav"):
                    filename = os.path.basename(file)
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(audio_output_dir, filename)

                    # Copy wav files to one output directory
                    if not os.path.exists(dest_file):
                        copyfile(src_file, dest_file)

        print("Speaker", speaker, "completed.")


def generate_world_features(filenames, data_dir, annotations_dict):
    """Code for creating and saving world features and sample labels"""

    world_dir = os.path.join(data_dir, 'world')
    f0_dir = os.path.join(data_dir, 'f0')
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(world_dir):
        os.mkdir(world_dir)
    if not os.path.exists(f0_dir):
        os.mkdir(f0_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    worlds_made = 0

    for i, f in enumerate(filenames):

        wav, labels = get_wav_and_labels(f, data_dir, annotations_dict)
        wav = np.array(wav, dtype=np.float64)
        labels = np.array(labels)

        filefront = f.split(".")[0]
        coded_sp_name = os.path.join(world_dir, filefront + ".npy")
        label_name = os.path.join(labels_dir, filefront + ".npy")
        f0_name = os.path.join(f0_dir, filefront + ".npy")
        if os.path.exists(coded_sp_name) and os.path.exists(label_name) and os.path.exists(f0_name):
            worlds_made += 1
            continue

        # Ignores data sample if wrong emotion
        if labels[0] != -1:
            f0, ap, sp, coded_sp = cal_mcep(wav)

            # Ignores data sample sample is too long
            if coded_sp.shape[1] < MAX_LENGTH:

                np.save(os.path.join(world_dir, filefront + ".npy"), coded_sp)
                np.save(os.path.join(labels_dir, filefront + ".npy"), labels)
                np.save(os.path.join(f0_dir, filefront + ".npy"), f0)

                worlds_made += 1
            else:
                print(f"Recording {filefront} too long, length {coded_sp.shape[1]}")
        # else:
        #     print(f"Recording {filefront} has invalid emotion {labels[0]}")

        if i % 10 == 0:
            print(i, " complete.")
            print(worlds_made, "worlds made.")


def generate_f0_stats(filenames, data_dir, annotations_dict):
    """Generate absolute and relative f0 dictionary"""

    FIRST_SPEAKER = 1
    NUM_SPEAKERS = 20
    NUM_EMOTIONS = 4
    f0_dir = os.path.join(data_dir, 'f0')

    # CALCULATE ABSOLUTE F0 STATS
    emo_stats = {}
    for e in range(NUM_EMOTIONS):
        spk_dict = {}
        for s in range(FIRST_SPEAKER, NUM_SPEAKERS + FIRST_SPEAKER):
            f0s = []
            for f in filenames:
                wav, labels = get_wav_and_labels(f, data_dir, annotations_dict)
                wav = np.array(wav, dtype=np.float64)
                labels = np.array(labels)
                if labels[0] == e and labels[1] == s:
                    f0_file = os.path.join(f0_dir, f[:-4] + ".npy")
                    if os.path.exists(f0_file):
                        f0 = np.load(f0_file)
                        f0s.append(f0)

            log_f0_mean, f0_std = get_f0_stats(f0s)
            spk_dict[s] = (log_f0_mean, f0_std)
            print(f"Done emotion {e}, speaker {s}.")
        emo_stats[e] = spk_dict

    with open('f0_dict.pkl', 'wb') as absolute_file:
        pickle.dump(emo_stats, absolute_file, pickle.HIGHEST_PROTOCOL)

    print(" ---- Absolute f0 stats completed ----")

    for tag, val in emo_stats.items():
        print(f'Emotion {tag} stats:')
        for tag2, val2 in val.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')

    # CALCULATE RELATIVE F0 STATS

    emo2emo_dict = {}

    for e1 in range(NUM_EMOTIONS):

        emo2emo_dict[e1] = {}

        for e2 in range(NUM_EMOTIONS):

            mean_list = []
            std_list = []

            for s in range(FIRST_SPEAKER, NUM_SPEAKERS + FIRST_SPEAKER):
                mean_diff = emo_stats[e2][s][0] - emo_stats[e1][s][0]
                std_diff = emo_stats[e2][s][1] - emo_stats[e1][s][1]
                mean_list.append(mean_diff)
                std_list.append(std_diff)

            mean_mean = np.mean(mean_list)
            std_mean = np.mean(std_list)
            emo2emo_dict[e1][e2] = (mean_mean, std_mean)

    print(" ---- Relative f0 stats completed ----")
    for tag, val in emo2emo_dict.items():
        print(f'Emotion {tag} stats:')
        for tag2, val2 in val.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')

    with open('f0_relative_dict.pkl', 'wb') as relative_file:
        pickle.dump(emo2emo_dict, relative_file, pickle.HIGHEST_PROTOCOL)


def run_preprocessing(args):

    print(f"--------------- Copying and restructuring IEMOCAP dataset in {args.data_dir} ---------------")
    copy_files(args.iemocap_dir, args.data_dir)

    data_dir = args.data_dir
    audio_dir = os.path.join(data_dir, 'audio')

    audio_filenames = [f for f in os.listdir(audio_dir) if '.wav' in f]

    annotations_dict = read_annotations(os.path.join(data_dir, 'annotations'))

    print("----------------- Producing WORLD features data -----------------")
    generate_world_features(audio_filenames, data_dir, annotations_dict)

    print("--------------- Producing relative f0 dictionaries ---------------")
    generate_f0_stats(audio_filenames, data_dir, annotations_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main preprocessing pipeline')
    parser.add_argument("--iemocap_dir", type=str, help="Directory of IEMOCAP dataset")
    parser.add_argument("--data_dir", type=str, default='./processed_data',
                        help="Directory to copy audio and annotation files to.")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    run_preprocessing(args)
