import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
# random.seed(2)
# torch.manual_seed(2)
# np.random.seed(2)
# Use GPU for training by default
cuda_flag = torch.cuda.is_available()
# device = torch.device("cpu")
device = torch.device("cuda:0" if cuda_flag else "cpu")
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Model arch name
model_arch_name = "resnet18" 
# Model number class
model_num_classes = 152
# Current configuration parameter method
mode = "train"


if mode == "train":
    
    image_size = 224
    batch_size = 40
    num_workers = 4

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 500

    # Loss parameters
    loss_label_smoothing = 0.0
    loss_aux3_weights = 1.0
    loss_aux2_weights = 0.3
    loss_aux1_weights = 0.3

    # How many iterations to print the training/validate result
    train_print_frequency = 50
    valid_print_frequency = 20

