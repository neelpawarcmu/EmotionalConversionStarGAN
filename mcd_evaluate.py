'''
mcd_evaluate.py

Author - Eric Zhou

Arguments:
    --checkpoint -c     : Checkpoint for conversion
'''

import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import math
import os
import pickle
import shutil
import pysptk

import librosa
from librosa.util import find_files
import pyworld
from pyworld import decode_spectral_envelope, synthesize

from matplotlib import pyplot as plt

import stargan.solver as solver
import stargan.model as model
import stargan.my_dataset as my_dataset
from stargan.my_dataset import get_filenames
from utils import audio_utils
import utils.data_preprocessing_utils as pp
import utils.preprocess_world as pw
from run_preprocessing import MAX_LENGTH

SAMPLING_RATE = 16000
FRAME_PERIOD = 5
FFT_SIZE = 1024


def mcd(ref_wav_file, synth_wav_file):
    def load_wav(wav_file, sr):
        wav, _ = librosa.load(wav_file, sr=sr, mono=True)
        return wav

    def log_spec_dB_dist(x, y):
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

    def wav_file_to_mcep(wav_file):
        wav = load_wav(wav_file, sr=SAMPLING_RATE)

        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=SAMPLING_RATE,
                                     frame_period=FRAME_PERIOD, fft_size=FFT_SIZE)

        # Extract MCEP features
        mcep_size = 34
        alpha = 0.35
        mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                               etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        return mgc

    ref_mgc = wav_file_to_mcep(ref_wav_file)
    synth_mgc = wav_file_to_mcep(synth_wav_file)

    min_cost, _ = librosa.sequence.dtw(ref_mgc[:, 1:].T, synth_mgc[:, 1:].T,
                                       metric=log_spec_dB_dist)

    dist = np.mean(min_cost) / len(ref_mgc)

    return dist

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint file of model')

    args = parser.parse_args()

    config = yaml.safe_load(open('./config.yaml', 'r'))

    args.out_dir = os.path.join(config['data']['dataset_dir'], 'converted')

    checkpoint_dir = args.checkpoint

    print("Loading model at ", checkpoint_dir)

    with open('neutral_mappings.pkl', 'rb') as f:
        neutral_to_emo_dict = pickle.load(f)

    print("Loaded neutral_mappings.pkl, mappings from neutral audio to emotional audio")

    #fix seeds to get consistent results
    SEED = 42
    # torch.backend.cudnn.deterministic = True
    # torch.backend.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Use GPU
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
        map_location='cuda'
    else:
        device = torch.device('cpu')
        map_location='cpu'

    # Load model
    model = model.StarGAN_emo_VC1(config, config['model']['name'])
    model.load(checkpoint_dir)
    config = model.config
    model.to_device(device = device)
    model.set_eval_mode()

    # Make emotion targets (using config file)
    # s = solver.Solver(None, None, config, load_dir = None)
    # targets =
    num_emos = config['model']['num_classes']
    emo_labels = torch.Tensor(range(0, num_emos)).long()
    emo_targets = F.one_hot(emo_labels, num_classes = num_emos).float().to(device = device)
    print(f"Number of emotions = {num_emos}")

    data_dir = config['data']['dataset_dir']
    annotations_dict = pp.read_annotations(os.path.join(data_dir, 'annotations'))

    # Convert only train and test samples
    data_dir = os.path.join(config['data']['dataset_dir'], 'audio')
    print("Converting train and test samples in", data_dir)

    ########################################
    #        WORLD CONVERSION LOOP         #
    ########################################
    train_npy_files, test_npy_files = my_dataset.get_train_test_split(
        os.path.join(config['data']['dataset_dir'], 'world'), config
    )

    def npy_to_wav(filename):
        """ Convert xxx.npy to xxx.wav.  """
        filefront = filename.split('.')[0]
        return filefront + ".wav"

    # Assuming that sample.npy in the world folder exists as sample.wav in the audio folder
    # train_wav_files = [npy_to_wav(f) for f in train_npy_files]
    test_wav_files = [npy_to_wav(f) for f in test_npy_files]

    def calculate_mcd(files, out_folder):
        """
        Params:
            - files: List of filenames
            - out_folder: Files get stored in <args.out_dir>/<out_folder>
        Returns nothing
        """
        total_distances = [0] * num_emos
        total_counts = [0] * num_emos
        for file_num, f in enumerate(files):
            filefront = os.path.basename(f)[:-4]
            input_wav_path = os.path.join(config['data']['dataset_dir'], 'audio', f)

            f = os.path.basename(f)[:-4] + ".wav"

            try:
                wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'], annotations_dict)
            except Exception as e:
                print("Warning:", e, "file", f)
                continue

            wav = np.array(wav, dtype = np.float64)
            labels = np.array(labels)

            # If wav file is not in original train / test input
            if labels[0] == -1 or labels[0] >= emo_targets.size(0):
                continue

            # Temporary: @eric-zhizu, if speaker is <= 10, it is Chinese so skip
            if labels[1] <= 10:
                continue

            f0_real, ap_real, sp, coded_sp = pw.cal_mcep(wav)

            # If in_dir not specified, and length of the recording >= MAX_LENGTH, skip
            if coded_sp.shape[1] >= MAX_LENGTH:
                continue

            # coded_sp_temp = np.copy(coded_sp).T
            # print(coded_sp_temp.shape)
            coded_sp_cpu = coded_sp.T
            coded_sp = torch.Tensor(coded_sp_cpu).unsqueeze(0).unsqueeze(0).to(device = device)

            with torch.no_grad():
                # print(emo_targets)
                for i in range (0, emo_targets.size(0)):
                    f0 = np.copy(f0_real)
                    ap = np.copy(ap_real)

                    f0 = audio_utils.f0_pitch_conversion(f0, (labels[0],labels[1]),
                                                             (i, labels[1]))

                    fake = model.G(coded_sp, emo_targets[i].unsqueeze(0))

                    ind2emo = {0: 'Neutral', 1: 'Happy', 2: 'Sad'}

                    if filefront in neutral_to_emo_dict \
                            and (ind2emo[i] in neutral_to_emo_dict[filefront] or i == 0):

                        if i != 0:
                            ref_wav_filefront = neutral_to_emo_dict[filefront][ind2emo[i]]
                            input_wav_path = os.path.join(data_dir, ref_wav_filefront + '.wav')

                        if not os.path.exists(input_wav_path):
                            print(f"{input_wav_path} does not exist")
                            continue

                        print(f"Converting {f[0:-4]} to {i}.")
                        model_iteration_string = model.config['model']['name'] + '_' + os.path.basename(args.checkpoint).replace('.ckpt', '')
                        filename_wav = model_iteration_string + '_' + f[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                    str(i) + ".wav"
                        filename_wav = os.path.join(args.out_dir, out_folder, filename_wav)

                        fake = fake.squeeze()
                        # print("Sampled size = ",fake.size())
                        # f = fake.data()
                        converted_sp = fake.cpu().numpy()
                        converted_sp = np.array(converted_sp, dtype = np.float64)

                        sample_length = converted_sp.shape[0]
                        if sample_length != ap.shape[0]:
                            # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[0:sample_length, :], dtype = np.float64)
                            ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
                            f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

                        f0 = np.ascontiguousarray(f0[20:-20], dtype = np.float64)
                        ap = np.ascontiguousarray(ap[20:-20,:], dtype = np.float64)
                        converted_sp = np.ascontiguousarray(converted_sp[20:-20,:], dtype = np.float64)
                        # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[40:-40,:], dtype = np.float64)

                        # print("ap shape = ", ap.shape)
                        # print("f0 shape = ", f0.shape)
                        # print(converted_sp.shape)
                        audio_utils.save_world_wav([f0,ap,sp,converted_sp], filename_wav)

                        dist = mcd(input_wav_path, filename_wav)

                        total_distances[i] += dist
                        total_counts[i] += 1

            # print(f, " converted.")
            if (file_num+1) % 20 == 0:
                print(file_num+1, " done.")
                print("Distances", total_distances)
                print("Counts", total_counts)

        mcd_per_emotion = [total_distances[emo] / total_counts[emo]
                           if total_counts[emo] != 0 else 0 for emo in range(len(total_counts))]
        return mcd_per_emotion

    print(calculate_mcd(test_wav_files[:500], "test"))

