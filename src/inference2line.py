'''
Plot prediction density with 2 lines as gnd , no decoding yet for prediction to the real lines
'''
import argparse
import os
from pathlib import Path

from torchvision.transforms import Resize, Normalize, Compose, CenterCrop, ToTensor

import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.stats import norm
from logging import getLogger
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
import tqdm
from matplotlib.pyplot import bar, figure, savefig, close, gca, legend

from utils import load_state_dict_resnet
import src.config as config
from src.functionals import leftright2normal, val2bin
from src.dataset import MEAN, STD
from src.train_HingeW import build_model

IMG_SIZE = 224
NUM_CLASS = 100

"""Logger for printing."""
_LOG = getLogger(__name__)


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    '''
    center crop of the image, resize, normalize, same as validation and test
    '''
    img = Image.open(image_path).convert('RGB')
    _LOG.info('Load image successfully')
    
    crop_size = min(img.size)
    transform = Compose([CenterCrop(crop_size),
                         Resize(IMG_SIZE),
                         ToTensor(),
                         Normalize(MEAN, STD)])
    pro_img = transform(img)
    # Transfer tensor channel image format data to CUDA device
    tensor = pro_img.unsqueeze(0)
    tensor = tensor.to(device=device)

    scale_factor = 224 / crop_size
    return tensor, img, scale_factor, crop_size


def load_annotation(image_path: Path, csv_path: Path):
    img_name = image_path.split('/')[-1]  
    data = pd.read_csv(csv_path)
    # data = deepcopy(data)
    row_idx = data[data["image_name"]== img_name].index.tolist()[0]
    row = data.iloc[row_idx]
    return row

def smooth_2gnd_Guassian(gnd_idx_1, gnd_idx_2, std: float, num_bins: int):
        '''
        turn one-hot gnd distribution into discrete Gussian distribution from [0, 99]
        or Gaussian + uniform distribution
        gnd_idx: a batch vector  
        '''
        batch_size = 1
        a = torch.arange(0, num_bins).unsqueeze(dim=1)
        b = torch.cat([a]*batch_size, axis=1).T
        std1 = torch.full(size=(batch_size,1), fill_value=float(std))
        dist = torch.from_numpy(norm.pdf(b, gnd_idx_1.unsqueeze(dim=1).cpu().numpy(), std1)) 
        
        non_inf_mask = 0
        num_non_inf = 1
        c = torch.cat([a]*num_non_inf, axis=1).T
        std2 = torch.full(size=(num_non_inf, 1), fill_value=float(std)) 
        dist += torch.from_numpy(norm.pdf(c, gnd_idx_2[non_inf_mask].cpu().numpy(), std2))
        
        normalized_dist = (dist.T / torch.sum(dist, dim=1)).T # sum to 1
        return normalized_dist.to(device=config.device, non_blocking=True)

def plot_density(gnd_idx_1, gnd_idx_2, dist, path):
    '''
    visualize distribution as a bar chart
    dist: distribution after softmax
    '''
    gnd_dist = smooth_2gnd_Guassian(gnd_idx_1, gnd_idx_2, 4, NUM_CLASS)
    gnd_dist = gnd_dist.cpu().numpy().squeeze()

    names = [str(i) for i in range(NUM_CLASS)]
    figure(figsize=(25, 25))
    bar(names, gnd_dist, alpha=0.5)
    bar(names, dist.squeeze(), alpha=0.7)   
    ax = gca()
    ax.axes.xaxis.set_ticklabels([])    
    legend(['GND', 'prediction'])
    savefig(path)
    close()
    _LOG.info(f'img {path} save successuflly')


def main():
    def get_prob(likelihood):
        prob = ((1e-7/NUM_CLASS+torch.nn.functional.softplus(likelihood)).T / (1e-7 + torch.sum(torch.nn.functional.softplus(likelihood), dim=1))).T
        return prob
    # load bin edges
    bin_edges = loadmat(args.bin_edges_path)

    device = choice_device(args.device_type)

    # Initialize the model
    resnet_model = build_model(pretrain=False)
    # print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _ = load_state_dict_resnet(resnet_model, args.checkpoint_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # load image paths
    with open(args.image_path) as f:
        image_path_list = f.read().splitlines() 

    count = 0
    for image_name in tqdm.tqdm(image_path_list):
        image_path = image_name               
        
        tensor, img, _, crop_size = preprocess_image(image_path, device)

        # Inference
        with torch.no_grad():
            output = resnet_model(tensor)
        output_rho = output[:, 0:NUM_CLASS] 
        output_theta = output[:, NUM_CLASS:] 
        prob_rho = get_prob(output_rho).cpu().numpy()
        prob_theta = get_prob(output_theta).cpu().numpy()   
            
        # #load gnd and scale to 224
        gnd_coord = load_annotation(image_path, args.csv_path)
        theta_1, rho_1 = leftright2normal(gnd_coord['left_y_1'], gnd_coord['left_x'], gnd_coord['right_y_1'], gnd_coord['right_x'], crop_size)
        # get bin idx
        theta_id_1, rho_id_1 = val2bin(theta_1, rho_1, bin_edges)
        if float(gnd_coord["left_y_2"]) == float('inf')\
            or float(gnd_coord["right_y_2"]) == float('inf'):
                theta_2, rho_2 = float('inf'), float('inf')
                theta_id_2, rho_id_2 = torch.tensor(float('inf')), torch.tensor(float('inf')) #float('inf'), float('inf')
        else:
            theta_2, rho_2 = leftright2normal(gnd_coord['left_y_2'], gnd_coord['left_x'], gnd_coord['right_y_2'], gnd_coord['right_x'], crop_size)
            theta_id_2, rho_id_2 = val2bin(theta_2, rho_2, bin_edges)
        
        # plot density
        plot_density(rho_id_1, rho_id_2, prob_rho, f'img/aleatoric/inference_{count}_rho.pdf')
        plot_density(theta_id_1, theta_id_2, prob_theta, f'img/aleatoric/inference_{count}_theta.pdf')
        count += 1
    return   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--csv_path", type=Path,
        default="data/hlw_2_0/metadata.csv",
        help="Path to the csv file")
    parser.add_argument("-b", '--bin_edges_path', type=Path, default='data/bins.mat')
    parser.add_argument("--checkpoint_path", type=Path, default="pretrained_model/renorm.02warmup.pth.tar")
    parser.add_argument("-i", "--image_path", type=Path, default="./data/inference.txt")
    parser.add_argument("--device_type", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--log_path", type=Path,
        default="logs/inference.log",
        help="Path to the training path file")
    args = parser.parse_args()

    main()
