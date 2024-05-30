'''
sparsification plot with an uncertainty measure
batch size is always 1
'''
import os
from pathlib import Path
import time
import datetime
from argparse import ArgumentParser

import pickle
import torch
from torch import nn
from logging import getLogger, basicConfig, INFO
from scipy.io import loadmat
from torch.distributions.categorical import Categorical
import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, figure, title, savefig, legend, axis, xlim

from torch.utils.data import DataLoader
from src.dataset import CUDAPrefetcher, TwoLineClassDataset
import src.config_test as config
from src.utils import load_state_dict_resnet, accuracy, Summary, AverageMeter, ProgressMeter, make_directory
from src.train_HingeW import InferenceOutputs
from src.functionals import bin_normal2leftright, normal2leftright
from src.train_HingeW import build_model
from src.test_metrics import get_horizon_error, get_AUC, Metrics, get_brier_ece
from src.layers import CRPS, EMDRenorm


NUM_CLASS = 100


"""Logger for printing."""
_LOG = getLogger(__name__)


def get_cdf(prob, gnd):
    """calculate the most probable bin's cdf, i.e., sum of all the bins on the left

    Args:
        prob (_type_): sum to 1, shape B*C
        gnd: ground truth bin id
    """
    return prob[:, :gnd+1].sum(axis=-1)


def get_prob(likelihood, flag:str):
    '''
    send likelihood to softplus and normalization to 1
    '''
    if flag == 'softplus':
        softplus = nn.Softplus()
        prob = (1e-7/NUM_CLASS+softplus(likelihood)) / (1e-7 + torch.sum(softplus(likelihood)))
    else:
        softmax = nn.Softmax()
        prob = softmax(likelihood)
    return prob

def get_std_idx(likelihood):
    '''
    in index space
    '''
    prob = get_prob(likelihood, args.activation).cpu().numpy()
    # print(prob.shape)
    idx = np.arange(0, prob.shape[0])
    mean = np.sum(prob*idx)
    var = np.sum(prob*idx*idx) -mean*mean    
    return var

def get_std_param(likelihood, bin_edge):
    '''
    standard deviation of softmax distribution
    param: dist: likelyhood before softmax
    '''
    prob = get_prob(likelihood, args.activation).cpu().numpy()
    # mean
    bin_edge[0] = 101*bin_edge[1] - 100*bin_edge[2]
    bin_edge[-1] = 101*bin_edge[-2] - 100*bin_edge[-3]
    bin_edge = np.array(bin_edge)
    mean =  np.sum(prob * (bin_edge[0:-1]+bin_edge[1:]))  / 2
    # variance
    mean_sq = np.sum(prob * (np.square(bin_edge[0:-1])+np.square(bin_edge[1:])+bin_edge[0:-1]*bin_edge[1:])) / 3
    var = mean_sq - mean*mean
    return var

def get_entropy_index(likelihood, args):
    '''
    param: dist: likelyhood before softplus
    '''
    prob = get_prob(likelihood, args.activation)
    entropy = Categorical(prob).entropy()
    return entropy.cpu().numpy()

def get_entropy_param(likelihood, bin_edges):
    '''
    param: dist: likelyhood before softplus
    '''
    bin_edges[0] = 101*bin_edges[1] - 100*bin_edges[2]
    bin_edges[-1] = 101*bin_edges[-2] - 100*bin_edges[-3]
    bin_edges = np.array(bin_edges)
    bin_width = bin_edges[1:] - bin_edges[0:-1]
    
    prob = get_prob(likelihood, args.activation).cpu().numpy()
    entropy = -np.sum(prob*np.log(prob/bin_width))
    return entropy

def get_L2(likelihood):
    '''
    see definition in (9)
    '''
    prob = get_prob(likelihood, args.activation).cpu()
    y_hat = prob.argmax()
    D2 = torch.square(y_hat-torch.arange(0, NUM_CLASS))
    return torch.sqrt(torch.sum(D2*prob)).cpu().numpy()

def get_L1(likelihood):
    '''
    see definition in (11)
    '''
    prob = get_prob(likelihood, args.activation).cpu()
    y_hat = prob.argmax()
    D = torch.abs(y_hat-torch.arange(0, NUM_CLASS))    
    return torch.sum(D*prob).numpy()

def get_auc(errors, fractions):
    '''
    for non-monotone curve, sklearn auc will not work. Here is a manual implement 
    parameter: errors: can be entropy/std, etc uncertainty metric
               fractions: np.arange(0, 1, 0.01)
    '''
    widths = np.abs(fractions[0:-1] - fractions[1:])
    ause = np.sum(errors[0:-1] * widths)
    return ause
    
def get_nll(outputs, gnd_id, batch_size:int):
    '''
    NLL in the index space?
    param: likelihood: likehood before normalization
    '''
    nll = 0
    for idx in range(batch_size):        
        likelihood = outputs[idx, :] # batch_size * categories
        prob = get_prob(likelihood, args.activation).cpu()
        nll -= np.log(prob[gnd_id])
    return  nll

def get_crps(outputs, gnd_id_1, gnd_id_2, batch_size:int, cost, idx:bool=True):
    gnd_id_1 = gnd_id_1.cpu().numpy()
    gnd_id_2 = gnd_id_2.cpu().numpy()
    emd = 0
    for ind in range(batch_size):
        emd += cost.forward(outputs, gnd_id_1[ind], gnd_id_2[ind], idx=idx)
    return emd
    
def get_batch_metrics(outputs, targets, bin_edges, batch_size:int, args)->list:
    '''ambiguity'''
    horizon_errors, theta_AE, rho_AE, theta_metrics, rho_metrics, joint_metrics = [], [], [], [], [], []
    for idx in range(batch_size):        
        output_theta_dist = outputs.theta[idx, :] # batch_size * categories
        output_rho_dist = outputs.rho[idx, :]
        if args.metric == 'entropy_index':
            theta_entropy = get_entropy_index(output_theta_dist, args)   
            rho_entropy = get_entropy_index(output_rho_dist, args)
        elif args.metric == 'entropy_param':
            theta_entropy = get_entropy_param(output_theta_dist, bin_edges['slope_bin_edges'])   
            rho_entropy = get_entropy_param(output_rho_dist, bin_edges['offset_bin_edges'])
        elif args.metric == 'L1':
            theta_entropy = get_L1(output_theta_dist)   
            rho_entropy = get_L1(output_rho_dist)
        elif args.metric == 'L2':
            theta_entropy = get_L2(output_theta_dist)   
            rho_entropy = get_L2(output_rho_dist)
        elif args.metric == 'std_index':
            theta_entropy = get_std_idx(output_theta_dist)   
            rho_entropy = get_std_idx(output_rho_dist)
        elif args.metric == 'std_param':
            theta_entropy = get_std_param(output_theta_dist, bin_edges['slope_bin_edges'])
            rho_entropy = get_std_param(output_rho_dist, bin_edges['offset_bin_edges'])
        else:
            raise('Wrong option')

        theta_metrics.append(theta_entropy)
        rho_metrics.append(rho_entropy)
        joint_metrics.append(theta_entropy+rho_entropy)
       
        target_theta = targets[0][idx].cpu().numpy()
        target_rho = targets[1][idx].cpu().numpy()
        # (theta, rho) ==> (left, right)
        output_left, output_right, output_theta, output_rho = bin_normal2leftright(output_theta_dist.argmax(), output_rho_dist.argmax(), bin_edges) 
        target_left, target_right = normal2leftright(target_theta, target_rho)
        # horrizon error
        horizon_error = get_horizon_error(target_left, target_right, output_left, output_right)
        horizon_errors.append(horizon_error)
        # absolute error for theta & rho
        theta_AE.append(np.abs(output_theta-target_theta))
        rho_AE.append(np.abs(output_rho-target_rho))
    return horizon_errors, theta_AE, rho_AE, theta_metrics, rho_metrics, joint_metrics


def load_dataset(csv_path, test_path, img_path, bin_edges:dict) -> CUDAPrefetcher:
    test_dataset = TwoLineClassDataset(mode='Valid', csv_path=csv_path, txt_path=test_path,
                                         img_path=img_path, bin_edges=bin_edges)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    return test_prefetcher

def get_bin_center(bin_edges):
    bin_center = 0.5 * (bin_edges[1:]+bin_edges[0:-1])
    bin_center[0] = bin_edges[1] # avoid infinity
    bin_center[-1] = bin_edges[-2]
    return bin_center

def get_sparsification_plot(list_AE:list, list_std:list, name:str, path, mode="test"):
    '''
    return AUSE
    '''
    num_samples = len(list_AE)
    list_AE = np.squeeze(np.array(list_AE))
    list_std = np.squeeze(np.array(list_std))
    max_MAE = np.mean(list_AE)
    sorted_list_AE = np.sort(list_AE)[::-1]
    sorted_idx_std = np.argsort(list_std)[::-1] # from high to small
    fractions = np.arange(0, 1, 0.01)
    MAE_GND = []
    MAE_std = []
    for frac in fractions:
        num_removed_samples = int(frac * num_samples)
        MAE_GND.append(np.mean(sorted_list_AE[num_removed_samples+1:])) # the mean AE of remaining samples
        MAE_std.append(np.mean(list_AE[sorted_idx_std[num_removed_samples+1:]]))  
        
    # for debug
    with open('./data/MAE_GND.txt', 'w') as f:
        f.writelines(str(MAE_GND))

    ause = get_auc(np.array(MAE_std)/max_MAE, fractions) - get_auc(np.array(MAE_GND)/max_MAE, fractions) 
    
    if mode == "test":
        figure()
        plot(fractions, MAE_GND/max_MAE)
        plot(fractions, MAE_std/max_MAE)
        ylabel('MAE')
        xlabel('Fraction of Removed Samples')
        xlim(0, 1)
        legend(['Oracle', 'Entropy'])
        axis('equal')
        title('AUSE:{:.1%}'.format(ause))
        
        make_directory(f'./img/{path}/')
        make_directory(f'./data/{path}/')
        savefig(f'./img/{path}/{name}_sparsification.pdf')
    
        # sparsification error
        error = (np.array(MAE_std)-np.array(MAE_GND))/max_MAE
        np.save(f'data/{path}/{name}_sparsification_error.npy', error)
    return ause

def main(args) -> None:
    start_time = datetime.datetime.now()
    _LOG.info(f"*** Test at {start_time.year}.{start_time.month}.{start_time.day} {start_time.hour}:{start_time.minute}:{start_time.second}***")
    
    bin_edges = loadmat(args.bin_edges_path)
    _LOG.info(f"Load bin edges successfully.")
    theta_centers = torch.from_numpy(get_bin_center(bin_edges['slope_bin_edges'])).to(device=config.device, non_blocking=True)
    rho_centers = torch.from_numpy(get_bin_center(bin_edges['offset_bin_edges'])).to(device=config.device, non_blocking=True)
    
    theta_crps = CRPS(thresh=0, bin_center=theta_centers)
    rho_crps = CRPS(thresh=0, bin_center=rho_centers)

    with open(args.checkpoint_path, 'r') as f:
        checkpoint_paths = f.read().splitlines()
    _LOG.info('Load checkpoints successfully.')

    # Load test dataloader
    test_prefetcher = load_dataset(args.csv_path, args.test_path, args.img_path, bin_edges)
    _LOG.info('Load dataset successfully.')
    
    # add hinge on pred
    hw = EMDRenorm(thresh=0.015)
    hw.batch_size = 1

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    theta_acc1 = AverageMeter("Acc@1", ":6.2f")
    theta_acc5 = AverageMeter("Acc@5", ":6.2f")
    rho_acc1 = AverageMeter("Acc@1", ":6.2f")
    rho_acc5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(batches, [batch_time, theta_acc1, 
                theta_acc5, rho_acc1, rho_acc5], prefix=f"Test: ")
    # initialize metrics
    metrics = Metrics()
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        detection_errors, theta_AEs, rho_AEs, theta_metrics, rho_metrics,\
            joint_metrics, crps_theta,crps_rho, nll_theta, nll_rho = [], [], [], [], [], [],  [], [], [], []
        theta_SE, rho_SE = 0, 0
        theta_cdfs, rho_cdfs = torch.zeros(1, device=config.device), torch.zeros(1, device=config.device)
        
        path = checkpoint_path.split('/')[5]  # fix this according to the path!!!!
        dataset = str(args.test_path).split('/')[-1].split('.')[0] # to specify it is the test set or validation set, training set
        _LOG.info(f'checkpoint:{checkpoint_path}\n dataset:{dataset}')
        
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

        # Initialize the data loader and load the first batch of data
        test_prefetcher.reset()
        batch_data = test_prefetcher.next()

        # Get the initialization test time
        end = time.time()

        with torch.no_grad():
            while batch_data is not None:
                # Transfer in-memory data to CUDA devices to stest_pathpeed up training
                images = batch_data[0].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
                theta_1_id = batch_data[1][0].to(device=config.device, non_blocking=True)
                rho_1_id = batch_data[1][1].to(device=config.device, non_blocking=True)
                theta_2_id = batch_data[2][0].to(device=config.device, non_blocking=True)
                rho_2_id = batch_data[2][1].to(device=config.device, non_blocking=True)
                theta = batch_data[3][0].to(device=config.device, non_blocking=True)
                rho = batch_data[3][1].to(device=config.device, non_blocking=True)
                
                # Get batch size
                batch_size = images.size(0)

                # Inferences                  
                output = resnet_model(images)    # batch_size * 100 classes
                output_rho = output[:, 0:NUM_CLASS]
                output_theta = output[:, NUM_CLASS:]
                output = InferenceOutputs(output_rho, output_theta)              

                # measure accuracy and record loss
                theta_top1, theta_top5 = accuracy(output.theta, theta_1_id, topk=(1, 5))  
                rho_top1, rho_top5 = accuracy(output.rho, rho_1_id, topk=(1, 5))  
                theta_acc1.update(theta_top1[0].item(), batch_size)
                theta_acc5.update(theta_top5[0].item(), batch_size)
                rho_acc1.update(rho_top1[0].item(), batch_size)
                rho_acc5.update(rho_top5[0].item(), batch_size)
                batch_horizon_errors, batch_theta_AE, batch_rho_AE, batch_theta_metrics, batch_rho_metrics, batch_joint_metrics = get_batch_metrics(outputs=output, targets=(theta, rho),  # change!!
                                                        batch_size=batch_size, bin_edges=bin_edges, args=args)
                
                # NLL 
                nll_theta.append(get_nll(output.theta, theta_1_id, batch_size).cpu().numpy())
                nll_rho.append(get_nll(output.rho, rho_1_id, batch_size).cpu().numpy())
                crps_theta.append(get_crps(output.theta, theta_1_id, theta_2_id, batch_size, theta_crps, idx=True).cpu().numpy()) 
                crps_rho.append(get_crps(output.rho, rho_1_id, rho_2_id, batch_size, rho_crps, idx=True).cpu().numpy())
                
                batch_theta_cdf = get_cdf(get_prob(output.theta, args.activation), theta_1_id)
                batch_rho_cdf = get_cdf(get_prob(output.rho, args.activation), rho_1_id)

                
                theta_cdfs = torch.cat((theta_cdfs, batch_theta_cdf), dim=-1)
                rho_cdfs = torch.cat((rho_cdfs, batch_rho_cdf), dim=-1)
                detection_errors += batch_horizon_errors
                theta_AEs += batch_theta_AE
                rho_AEs += batch_rho_AE
                theta_metrics += batch_theta_metrics 
                rho_metrics += batch_rho_metrics
                joint_metrics += batch_joint_metrics
                
                theta_SE += np.sum(np.array(batch_theta_AE) * np.array(batch_theta_AE)) 
                rho_SE += np.sum(np.array(batch_rho_AE) * np.array(batch_rho_AE))
                
                # Calculate the time it takes to fully train a batch of data
                batch_time.update(time.time() - end)
                end = time.time()

                # Write the data during training to the training log file
                if batch_index % config.test_print_frequency == 0:
                    progress.display(batch_index + 1)

                # Preload the next batch of data
                batch_data = test_prefetcher.next()

                # Add 1 to the number of data batches to ensure that the terminal prints data normally
                batch_index += 1

        # print metrics
        theta_RMSE = np.sqrt(theta_SE/batch_index)
        rho_RMSE = np.sqrt(rho_SE/batch_index)
        crps_theta = np.array(crps_theta).mean()
        crps_rho = np.array(crps_rho).mean()
        nll_theta = np.array(nll_theta).mean()
        nll_rho = np.array(nll_rho).mean()
        theta_cdfs = theta_cdfs[1:]
        rho_cdfs = rho_cdfs[1:]

        _LOG.info(f"{path}")
        theta_ece, _ = get_brier_ece(theta_cdfs, vis=True, savefig_filename=f'./img/theta_{path}_calibration.pdf')
        rho_ece, _ = get_brier_ece(rho_cdfs, vis=True, savefig_filename=f'./img/rho_{path}_calibration.pdf')
        _LOG.info("CALIBRATION")


        auc = get_AUC(detection_errors)
        entries = progress.display_summary()
        _LOG.info("\t".join(entries))
        _LOG.info(f"AUC {auc*4}")
        _LOG.info(f"Calibration theta {theta_ece} rho {rho_ece}")
        _LOG.info('RMSE theta {} rho {} \n CRPS theta {} rho {} \n NLL theta {} rho {}'.format(theta_RMSE, rho_RMSE, crps_theta, crps_rho, nll_theta, nll_rho))
        
        # save error
        result = {'theta AE':theta_AEs, 'theta metrics':theta_metrics, 'rho AE':rho_AEs, 'rho metrics':rho_metrics, 'horizon error':detection_errors, 'joint entropy': joint_metrics}
        make_directory(f'./data/{path}')
        with open(f'./data/{path}/resnet_{args.metric}_index_{dataset}.pickle', 'wb') as f: 
            pickle.dump(result, f)
        
        #sparsification plot
        theta_ause = get_sparsification_plot(theta_AEs, theta_metrics, name=f'theta_{args.metric}_{path}_{dataset}', path=path)
        rho_ause = get_sparsification_plot(rho_AEs, rho_metrics, name=f'rho_{args.metric}_{path}_{dataset}', path=path)
        joint_ause = get_sparsification_plot(detection_errors, joint_metrics, name=f'joint_{args.metric}_{path}_{dataset}', path=path)
        _LOG.info(f'theta AUSE {theta_ause} rho AUSE {rho_ause} joint AUSE {joint_ause}')

        metrics.update((theta_RMSE, crps_theta, nll_theta, theta_ause, theta_ece), 'theta')
        metrics.update((rho_RMSE, crps_rho, nll_rho, rho_ause, rho_ece), 'rho')
        metrics.update((auc, joint_ause), 'joint')
    
    result = metrics.get_mean_std()
    _LOG.info(result)
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Test")
    parser.add_argument(
        "-c", "--csv_path", type=Path,
        default="/data/datasets/HLW2/metadata.csv",
        help="Path to the csv file")
    parser.add_argument(
        "-i", "--img_path", type=Path,
        default="/data/datasets/HLW2/images",
        help="Path to the images")
    parser.add_argument(
        "-t", "--test_path", type=Path,
        default="/data/datasets/HLW2/split/test.txt",
        help="Path to the test path file")
    parser.add_argument(
        "-b", "--bin_edges_path", type=Path,
        default="data/bins.mat",
        help="Path to the training path file")
    parser.add_argument(
        "--checkpoint_path", type=Path,
        default="data/ensemble_checkpoints.txt",
        help="Path to the training path file")
    parser.add_argument(
        "--log_path", type=Path,
        default="logs/test.log",
        help="Path to the training path file")
    parser.add_argument(
        "--activation", type=str,
        choices=['softplus', 'softmax'],
        default='softplus')
    parser.add_argument(
        "--metric", type=str,
        choices=['entropy_index', 'entropy_param', 'std_index', 'std_param', 'L1', 'L2'],
        default='entropy_index')
    args = parser.parse_args()
    basicConfig(filename=args.log_path, level=INFO)
    main(args)
