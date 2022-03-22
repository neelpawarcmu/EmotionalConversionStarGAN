"""
main.py

Authors - Max Elliott, Eric Zhou

Main script to start training of the proposed model (StarGAN_emo_VC1).

Command line arguments:

    --name -n       : Change the config[model][name] to --name if desired
    --checkpoint -c : Directory of checkpoint to resume training from if desired
    --load_emo      : Directory of pretrained emotional classifier checkpoint to
                      use if desired
    --evaluate -e   : Flag to run model in test mode rather than training
    --alter -a      : Flag to change the config file of a loaded checkpoint to
                      ./config.yaml (this is useful if models are trained in
                      stages as propsed in the project report)

"""

import argparse
import torch
import yaml
import numpy as np
import random
import os

import stargan.my_dataset as my_dataset
from stargan.my_dataset import get_filenames
import stargan.solver as solver

def make_weight_vector(filenames, data_dir):

    label_dir = os.path.join(data_dir, 'labels')

    emo_labels = []

    for f in filenames:

        label = np.load(label_dir + "/" + f + ".npy")
        emo_labels.append(label[0])

    categories = list(set(emo_labels))
    total = len(emo_labels)

    counts = [total/emo_labels.count(emo) for emo in range(len(categories))]

    weights = torch.Tensor(counts)

    return weights

    # self.emo_loss_weights = torch.Tensor([4040./549, 4040./890,
    #                                          4040./996, 4040./1605]).to(self.device)


if __name__ == '__main__':

    # ADD ALL CONFIG ARGS
    parser = argparse.ArgumentParser(description='StarGAN-emo-VC main method')
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Model name for training.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Directory of checkpoint to resume training from")
    parser.add_argument("--recon_only", help='Train a model without auxiliary classifier: learn to reconstruct input.',
                        action='store_true')
    parser.add_argument("--config", help='path of config file for training. Defaults = ./config.yaml',
                        default='./config.yaml')
    parser.add_argument("-a", "--alter", action='store_true')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    if args.name is not None:
        config['model']['name'] = args.name
        print(config['model']['name'])

    # fix seeds to get consistent results
    # torch.backend.cudnn.deterministic = True
    # torch.backend.cudnn.benchmark = False
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Use GPU
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
    else:
        device = torch.device('cpu')

    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.cuda)

    # Get correct data directory depending on features being used
    if config['data']['type'] == 'world':
        print("Using WORLD features.")
        assert config['model']['num_feats'] == 36

        data_dir = os.path.join(config['data']['dataset_dir'], "world")
    else:
        print("Using mel spectrograms.")
        assert config['model']['num_feats'] == 80

        data_dir = os.path.join(config['data']['dataset_dir'], "mels")

    print("Data directory = ", data_dir)

    # MAKE TRAIN + TEST SPLIT
    train_files, test_files = my_dataset.get_train_test_split(data_dir, config)

    print("Train test split")

    # @eric-zhizu: Not sure why original code makes weight vectors using both train and test files
    weight_vector = make_weight_vector(train_files + test_files, config['data']['dataset_dir'])

    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    # print(train_files[0:20])
    # print(test_files)

    # print(np.load(data_dir + "/" + files[0] + ".npy").shape)

    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)

    batch_size = config['model']['batch_size']

    train_loader, test_loader = my_dataset.make_variable_dataloader(train_dataset,
                                                                    test_dataset,
                                                                    batch_size=batch_size)

    print("Performing whole network training.")
    s = solver.Solver(train_loader, test_loader, config, load_dir=args.checkpoint, recon_only=args.recon_only)

    s.model.load_pretrained_classifier()
    print("Loaded pre-trained emotional classifier.")

    if args.alter:
        print(f"Changing loaded config to new config {args.config}.")
        s.config = config
        s.set_configuration()

    s.set_classification_weights(weight_vector)

    print("Training model.")
    s.train()
