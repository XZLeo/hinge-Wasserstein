'''
Model: Resnet
Scheduler: Linear
Loss: HingeEMD
gnd: Gasuusian
'''
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import time
import datetime
from logging import getLogger, basicConfig, INFO
from scipy.io import loadmat
import wandb
from collections import namedtuple
from typing import Optional

import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import models

import src.config as config
from src.dataset import CUDAPrefetcher
from src.dataset import ClassificationDataset
from src.utils import accuracy, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter
from src.layers import EMDRenorm
from src.test_metrics import get_AUC, get_batch_horizon_error


InferenceOutputs = namedtuple("GoogLeNetOutputs", ["rho", "theta"]) 
InferenceOutputs.__annotations__ = {"rho": Tensor, "theta": Optional[Tensor]}

NUM_CLASS = 100

"""Logger for printing."""
_LOG = getLogger(__name__)

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
                                            args.img_path, bin_edges, args.flipThresh)
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
        save_checkpoint(epoch,
                        {"epoch": epoch + 1,
                         "best_auc": best_auc,
                         "state_dict": resnet_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        is_best,
                        is_last)


def load_dataset(csv_path, train_path, valid_path, img_path, bin_edges:dict, flip_thresh:float=0.5): 
    # Load train, test and valid datasets
    train_dataset = ClassificationDataset(mode='Train', csv_path=csv_path, txt_path=train_path, 
                                         img_path=img_path, bin_edges=bin_edges, flip_thresh=flip_thresh)
    valid_dataset = ClassificationDataset(mode='Valid', csv_path=csv_path, txt_path=valid_path,
                                         img_path=img_path, bin_edges=bin_edges, flip_thresh=flip_thresh)

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    
    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    return train_prefetcher, valid_prefetcher


def build_model(pretrain:bool):  
    # resnet_model = models.resnet34(pretrained=pretrain)    
    resnet_model = models.resnet18(pretrained=pretrain)    
    resnet_model.fc = nn.Linear(512, NUM_CLASS*2)                                    
    # resnet_model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.2, training=m.training))                       
    resnet_model = resnet_model.to(device=config.device, memory_format=torch.channels_last)
    return resnet_model


def define_loss():
    '''
    Choose smoothing method and normalization
    '''
    criterion = EMDRenorm(args.renormThresh)
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)
    return criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.AdamW(model.parameters(),
                            lr=2e-4,
                            weight_decay=1e-4)
    return optimizer


def define_scheduler(optimizer: optim.SGD):
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    return scheduler


def train(
        model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        cuda_flag: False,
        theta_criterion: nn.CrossEntropyLoss,
        rho_criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    theta_loss = AverageMeter("Theta Loss", ":6.6f")
    rho_loss = AverageMeter("Rho Loss", ":6.6f")
    theta_acc1 = AverageMeter("Theta Acc@1", ":6.2f")
    theta_acc5 = AverageMeter("Theta Acc@5", ":6.2f")
    rho_acc1 = AverageMeter("Rho Acc@1", ":6.2f")
    rho_acc5 = AverageMeter("Rho Acc@5", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, theta_loss, rho_loss,
                              theta_acc1, theta_acc5, rho_acc1, rho_acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    if cuda_flag:
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()
    else:
        train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data[0].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        theta_id = batch_data[1][0].to(device=config.device, non_blocking=True)
        rho_id = batch_data[1][1].to(device=config.device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images.float())
            output_rho = output[:, 0:NUM_CLASS]
            output_theta = output[:, NUM_CLASS:]
            
            loss_rho = config.loss_aux3_weights * rho_criterion(output_rho, rho_id)
            loss_theta = config.loss_aux3_weights * theta_criterion(output_theta, theta_id) 
            loss = loss_rho + loss_theta
                        
        # Backpropagation        
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss 
        theta_top1, theta_top5 = accuracy(output_theta, theta_id, topk=(1, 5))  
        rho_top1, rho_top5 = accuracy(output_rho, rho_id, topk=(1, 5))  
        theta_loss.update(loss_theta.item(), batch_size)
        rho_loss.update(loss_rho.item(), batch_size)
        theta_acc1.update(theta_top1[0].item(), batch_size)
        theta_acc5.update(theta_top5[0].item(), batch_size)
        rho_acc1.update(rho_top1[0].item(), batch_size)
        rho_acc5.update(rho_top5[0].item(), batch_size)


        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            wandb.log({"Train/Loss":loss.item(), "Train/Theta Loss":loss_theta.item(),
            "Train/Rho Loss": loss_rho.item(),
            "Train/Theta Accuracy@1": theta_acc1.avg, 
            "Train/Rho Accuracy@1": rho_acc1.avg})  
            entries = progress.display(batch_index + 1)
            _LOG.info("\t".join(entries))

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1
    entries = progress.display_summary()
    _LOG.info("\t".join(entries))
    return


def validate(
        ema_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,        
        mode: str,
        bin_edges: dict
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    theta_acc1 = AverageMeter("Theta Acc@1", ":6.2f")
    theta_acc5 = AverageMeter("Theta Acc@5", ":6.2f")
    rho_acc1 = AverageMeter("Rho Acc@1", ":6.2f")
    rho_acc5 = AverageMeter("Rho Acc@5", ":6.2f")
    progress = ProgressMeter(batches, [batch_time, theta_acc1, 
        theta_acc5, rho_acc1, rho_acc5], prefix=f"{mode}: ")
    detection_errors = []

    # Put the exponential moving average model in the verification mode
    ema_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data[0].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            theta_id = batch_data[1][0].to(device=config.device, non_blocking=True)
            rho_id = batch_data[1][1].to(device=config.device, non_blocking=True)
            theta = batch_data[2][0].to(device=config.device, non_blocking=True)
            rho = batch_data[2][1].to(device=config.device, non_blocking=True)
            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = ema_model(images.float())  # dimension [40, 100] * 2
            output_rho = output[:, 0:NUM_CLASS]
            output_theta = output[:, NUM_CLASS:]

            # measure accuracy and record loss
            theta_top1, theta_top5 = accuracy(output_theta, theta_id, topk=(1, 5))  
            rho_top1, rho_top5 = accuracy(output_rho, rho_id, topk=(1, 5))      
            theta_acc1.update(theta_top1[0].item(), batch_size)
            theta_acc5.update(theta_top5[0].item(), batch_size)
            rho_acc1.update(rho_top1[0].item(), batch_size)
            rho_acc5.update(rho_top5[0].item(), batch_size)

            output_renamed = InferenceOutputs(output_rho, output_theta)
            # output_renamed = InferenceOutputs(rho_id, theta_id) # use GND to test quantization error
            detection_errors += get_batch_horizon_error(outputs=output_renamed, targets=(theta, rho),
                                                     batch_size=batch_size, bin_edges=bin_edges)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                entries = progress.display(batch_index + 1)
                _LOG.info("\t".join(entries))

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    entries = progress.display_summary()
    _LOG.info("\t".join(entries))

    AUC = get_AUC(detection_errors)

    if mode == "Valid" or mode == "Test":
        wandb.log({"Theta Acc@1": theta_acc1.avg,  
                "Rho Acc@1": rho_acc1.avg, 
                "AUC": AUC})
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return AUC


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
        "--renormThresh", type=float,
        default=0.01,
        help='threshold for renormalization')
    parser.add_argument(
        "--flipThresh", type=float,
        default=0.5,
        help="0.5 for random flip, 0 for no flip, 10 for always flip")
    args = parser.parse_args()
    log_path = 'logs/{}.log'.format(args.training_name)
    with open(log_path, 'w') as fp:
        pass    
    basicConfig(filename=log_path, level=INFO)
    main(args)


