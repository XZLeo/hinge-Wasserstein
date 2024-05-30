'''
Visualize the density estimation of a trained model in the sequence of uncertainty metrics, e.g., entropy
1. Run test_resnet.py to resnet_entropy.pickle with an uncertainty measure
2. Run this file to get density estimation, sorted by the uncertainty measure, from small to high or vice versa
'''
import os
from pathlib import Path
from argparse import ArgumentParser

import pickle
import torch
from torch.utils.data import DataLoader
from logging import getLogger, basicConfig, INFO
import numpy as np
from scipy.io import loadmat

import src.config as config
from src.utils import load_state_dict_resnet, make_directory
from src.dataset import TwoLineClassDataset, CUDAPrefetcher
from src.test import get_prob
from src.train_HingeW import InferenceOutputs, build_model
from src.inference2line import plot_density

NUM_CLASSES = 100

"""Logger for printing."""
_LOG = getLogger(__name__)
    
def load_dataset(csv_path, test_path, img_path, bin_edges:dict):
    test_dataset = TwoLineClassDataset(mode='Valid', csv_path=csv_path, txt_path=test_path,
                                         img_path=img_path, bin_edges=bin_edges)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    return test_prefetcher



def main() -> None:
    start = 900
    bin_edges = loadmat(args.bin_edges_path)
    _LOG.info(f"Load bin edges successfully.")
    
    with open(args.checkpoint_path, 'r') as f:
        checkpoint_paths = f.read().splitlines()
    _LOG.info('Load checkpoints paths successfully.')

    
    # Load test dataloader
    test_prefetcher = load_dataset(args.csv_path, args.test_path, args.img_path, bin_edges)
    test_path = str(args.test_path).split('/')[3].split('.')[0]
    _LOG.info('Load dataset successfully.')
    
    # initialize all ensembles
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        path = checkpoint_path.split('/')[4]
        metric_path = f'./data/{path}/resnet_entropy_index_index_{test_path}.pickle' # change it to args.test_path
        with open(metric_path, 'rb') as f:
                result = pickle.load(f)
        _LOG.info("Load metrics successfully.")
        
        # sort std, find img idx
        theta_std_list = result['theta metrics']        
        rho_std_list = result['rho metrics']
        sorted_theta_std_idx = np.argsort(theta_std_list)[start:start+50] # smallest 10  # entropy is -sum(P*log(P))
        _LOG.info(f'{sorted_theta_std_idx}')

        # Initialize the model
        resnet_model = build_model(pretrain=False)  
        _LOG.info(f"Build the {idx} `{config.model_arch_name}` model successfully.")

        # Load model weights
        resnet_model, _, _, _, _ = load_state_dict_resnet(resnet_model, checkpoint_path)
        _LOG.info(f"Load `{config.model_arch_name}` "
            f"model weights `{os.path.abspath(checkpoint_path)}` successfully.")
         # Start the verification mode of the model.
        resnet_model.eval()    

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0
        plot_index = 0

        # Initialize the data loader and load the first batch of data
        test_prefetcher.reset()
        batch_data = test_prefetcher.next()

        with torch.no_grad():
            # plot according to theta's sequence
            # batch_index is the img's index in the test set
            while batch_data is not None:
                if batch_index not in sorted_theta_std_idx: 
                    batch_data = test_prefetcher.next()
                    batch_index += 1
                    continue
                _LOG.info('Find one image')
                a = list(sorted_theta_std_idx)
                plot_index = start + a.index(batch_index) # image {batch_index}'s sequence in terms of an uncertainty measure
                
                # Transfer in-memory data to CUDA devices to speed up training
                images = batch_data[0].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
                theta_1_id = batch_data[1][0].to(device=config.device, non_blocking=True)
                rho_1_id = batch_data[1][1].to(device=config.device, non_blocking=True)
                theta_2_id = batch_data[2][0].to(device=config.device, non_blocking=True)
                rho_2_id = batch_data[2][1].to(device=config.device, non_blocking=True)

                # Inference
                output = resnet_model(images)
                _LOG.info('Inference successfully')
                output_rho = output[:, 0:NUM_CLASSES]
                output_theta = output[:, NUM_CLASSES:]
                output = InferenceOutputs(output_rho, output_theta)
                
                # softmax of the average
                # output = InferenceOutputs(output_rho, output_theta)
                
                img = f'2lines_{3999+batch_index}.png'
                make_directory(f'./img/density/{path}_{test_path}/')
                plot_density(theta_1_id, theta_2_id, get_prob(output.theta, args.activation).cpu().numpy(), f'./img/density/{path}_{test_path}/{plot_index}_theta_{img}.pdf')
                plot_density(rho_1_id, rho_2_id, get_prob(output.rho, args.activation).cpu().numpy(), f'./img/density/{path}_{test_path}/{plot_index}_rho_{img}.pdf')

                # Preload the next batch of data
                batch_data = test_prefetcher.next()

                # Add 1 to the number of data batches to ensure that the terminal prints data normally
                batch_index += 1
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Test")
    parser.add_argument(
        "-c", "--csv_path", type=Path,
        default="/mimer/NOBACKUP/groups/alvis_cvl/ziliang/hlw/data/hlw_2_0/metadata.csv",
        help="Path to the csv file")
    parser.add_argument(
        "-i", "--img_path", type=Path,
        default="/mimer/NOBACKUP/groups/alvis_cvl/ziliang/hlw/data/hlw_2_0/images",
        help="Path to the images")
    parser.add_argument(
        "-t", "--test_path", type=Path,
        default="/mimer/NOBACKUP/groups/alvis_cvl/ziliang/hlw/data/hlw_2_0/split/test.txt",
        help="Path to the test path file")
    parser.add_argument(
        "-b", "--bin_edges_path", type=Path,
        default="data/bins.mat",
        help="Path to the training path file")
    parser.add_argument(
        "--checkpoint_path", type=Path,
        default="data/resnet.txt",
        help="Path to the training path file") # read a text file contains checkpoints
    parser.add_argument(
        "--log_path", type=Path,
        default="logs/test_ensemble.log",
        help="Path to the training path file")
    parser.add_argument(
        "--activation", type=str,
        choices=['softplus', 'softmax'],
        default='softplus')
    parser.add_argument(
        "--metric_path", type=Path,
        default="./data/EMD/resnet_entropy_test.pickle",
        help="Path to the training path file")
    args = parser.parse_args()
    basicConfig(filename=args.log_path, level=INFO)
    main()
