'''
Plot gnd with prediction
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
from logging import getLogger
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
import tqdm
from matplotlib.pyplot import bar, ylabel, figure, savefig, close, gca

from utils import load_state_dict_resnet
from src.functionals import normal2leftright, leftright2normal, bin_normal2leftright
from src.dataset import MEAN, STD
from src.plot_horizon import plot_centerCrop_gnd_output, plot_gnd
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
    tensor = transform(img)
    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device=device)

    scale_factor = 224 / crop_size
    return tensor, img, scale_factor, crop_size


def load_annotation(image_path: Path, csv_path: Path):
    img_name = os.path.join(image_path.split('/')[-2], image_path.split('/')[-1]) 
    data = pd.read_csv(csv_path)
    # data = deepcopy(data)
    row_idx = data[data["image_name"]== img_name].index.tolist()[0]
    row = data.iloc[row_idx]
    return np.array((row["left_y"], row["left_x"], row["right_y"], row["right_x"]), dtype=float)


def plot_density(dist, path):
    '''
    visualize distribution as a bar chart
    dist: distribution after softmax/softplus
    '''
    num_bins = dist.shape[1]
    names = [str(i) for i in range(num_bins)]
    figure(figsize=(25, 25))
    bar(names, dist.squeeze())
    # xlabel('bins')
    ylabel('Density')
    ax = gca()
    ax.axes.xaxis.set_ticklabels([])    
    # ax.axes.yaxis.set_ticklabels([])
    # title(path)
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
    resnet_model = build_model(False)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights 
    resnet_model, _, _, _, _ = load_state_dict_resnet(resnet_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # load image paths
    with open(args.image_path) as f:
        image_path_list = f.read().splitlines() 

    count = 0
    for image_path in tqdm.tqdm(image_path_list):
        image_path = '/data/datasets/HLW2/images/' + image_path      
        
        tensor, img, _, crop_size = preprocess_image(image_path, device)

        # Inference
        with torch.no_grad():
            output = resnet_model(tensor)
        output_rho = output[:, 0:NUM_CLASS] 
        output_theta = output[:, NUM_CLASS:] 
        prob_rho = get_prob(output_rho).cpu().numpy()
        prob_theta = get_prob(output_theta).cpu().numpy()   
        
        # plot density
        plot_density(prob_rho, f'img/inference_{count}_theta.jpg')
        plot_density(prob_theta, f'img/inference_{count}_rho.jpg')
        count += 1
            
        output_left, output_right, _, _ = bin_normal2leftright(prob_theta.argmax(), prob_rho.argmax(), bin_edges)
    
        #load gnd and scale to 224
        gnd_coord = load_annotation(image_path, args.csv_path)
        theta, rho = leftright2normal(gnd_coord[0], gnd_coord[1], gnd_coord[2], gnd_coord[3], crop_size)
        gnd_left_y, gnd_right_y = normal2leftright(theta, rho, caffe_sz=IMG_SIZE)
        resized_gnd_coord = (gnd_left_y, -IMG_SIZE/2, gnd_right_y, IMG_SIZE/2)
        pred_coord = (output_left, -IMG_SIZE/2, output_right, IMG_SIZE/2)
        #visualization: plot gnd and output 
        file_name = image_path.split('/')[-1].split('.')[0]

        plot_gnd(img, gnd_coord[0], gnd_coord[1], gnd_coord[2], gnd_coord[3], file_name+'_gnd')
        plot_centerCrop_gnd_output(tensor, pred_coord, resized_gnd_coord, file_name)
    return   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--csv_path", type=Path,
        default="data/hlw_2_0/metadata.csv",
        help="Path to the csv file")
    parser.add_argument('--bin_edges_path', type=Path, default='data/bins.mat')
    parser.add_argument("--model_weights_path", type=Path, default="pretrained_model/renorm.02warmup.pth.tar")
    parser.add_argument("--image_path", type=Path, default="./data/inference.txt")
    parser.add_argument("--device_type", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--log_path", type=Path,
        default="logs/inference.log",
        help="Path to the training path file")
    args = parser.parse_args()

    main()
