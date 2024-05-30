'''
ResNet+Softmax-cross-entropy loss
'''
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import datetime
from logging import getLogger, basicConfig, INFO
from scipy.io import loadmat

import torch
from torch import nn
from torch.cuda import amp
import wandb

import config
from src.utils import make_directory, save_checkpoint
from src.train_HingeW import load_dataset, build_model, define_optimizer, define_scheduler, train, validate


NUM_CLASS = 100

"""Logger for printing."""
_LOG = getLogger(__name__)


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)
    return criterion

def main(args: Namespace)->None:
    time = datetime.datetime.now()
    _LOG.info(f"***{args.training_name} Training begins at {time.year}.{time.month}.{time.day} {time.hour}:{time.minute}:{time.second}***")
    
    for k,v in vars(args).items():
        _LOG.info(k + ": " + str(args.__dict__[k]))
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_auc = 0.0

    bin_edges = loadmat(args.bin_edges_path)
    _LOG.info(f"Load bin edges successfully.")
    
    train_prefetcher, valid_prefetcher = load_dataset(args.csv_path, args.train_path, args.validation_path, 
                                            args.img_path, bin_edges)
    _LOG.info(f"Load datasets successfully.")

    resnet_model = build_model(pretrain=True)
    _LOG.info("Build model successfully")

    theta_criterion = define_loss()
    rho_criterion = define_loss()
    _LOG.info("Define all loss functions successfully.")

    optimizer = define_optimizer(resnet_model)
    _LOG.info("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    _LOG.info("Define all optimizer scheduler functions successfully.")
    
    wandb.init(name=args.training_name, project="HLW")
    wandb.config = {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 128
            }

    # Create a experiment results
    samples_dir = os.path.join(args.checkpoint_path, "samples", args.training_name)
    results_dir = os.path.join(args.checkpoint_path, "results", args.training_name)
    make_directory(samples_dir)
    make_directory(results_dir)
    
    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(resnet_model, train_prefetcher, 
              config.cuda_flag, theta_criterion, 
              rho_criterion, optimizer, epoch, scaler)
        auc = validate(resnet_model, valid_prefetcher, epoch, "Valid", bin_edges)
        _LOG.info("AUC:{}\n".format(auc))
        scheduler.step()
        
        # Automatically save the model with the highest index
        is_best = auc > best_auc
        is_last = (epoch + 1) == config.epochs
        best_auc = max(auc, best_auc)
        save_checkpoint({"epoch": epoch + 1,
                         "best_auc": best_auc,
                         "state_dict": resnet_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        is_best,
                        is_last)



if __name__ == "__main__":
    parser = ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument(
        "-c", "--csv_path", type=Path,
        default="data/hlw_1_2/metadata.csv",
        help="Path to the csv file")
    parser.add_argument(
        "-i", "--img_path", type=Path,
        default="data/hlw_1_2/images",
        help="Path to the images")
    parser.add_argument(
        "-t", "--train_path", type=Path,
        default="data/hlw_1_2/split/train.txt",
        help="Path to the training path file")
    parser.add_argument(
        "-v", "--validation_path", type=Path,
        default="data/hlw_1_2/split/val.txt",
        help="Path to the training path file")
    parser.add_argument(
        "-b", "--bin_edges_path", type=Path,
        default="data/bin_edges",
        help="Path to the training path file")
    parser.add_argument(
        "--checkpoint_path", type=Path,
        default="/work2/hlw",
        help="Path to the training path file")
    parser.add_argument(
        "--training_name", type=str,
        help="Describe the training")
    parser.add_argument(
        "--flip_thresh", type=float,
        default=0.5,
        help="0.5 for random flip, 0 for no flip, 10 for always flip")
    args = parser.parse_args()
    log_path = 'logs/{}.log'.format(args.training_name)
    with open(log_path, 'w') as fp:
        pass    
    basicConfig(filename=log_path, level=INFO)
    main(args)

