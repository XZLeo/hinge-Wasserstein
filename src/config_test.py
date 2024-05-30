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
mode = "test"


if mode == "test":
    # Test dataloader parameters
    image_size = 224
    batch_size = 1
    num_workers = 4

    # How many iterations to print the testing result
    test_print_frequency = 20