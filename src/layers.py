import torch
import torch.nn as nn
import numpy as np
import src.config as config
from scipy.stats import norm
from torch.distributions import Categorical


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)
    
class ABS(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    
class Logistic(nn.Module):
    def forward(self, x):
        return torch.special.expit(x)
    

class EMDClose(nn.Module):
    '''
    closed form solution of Wasserstein distance in the case of 1D distribution
    '''
    def __init__(self, norm:str, smooth:str, smooth_coefficient:float) -> None:
        '''
        param norm: 'softplus', 'relu', 'square', 'abs', 'softmax'
              beta: for softplus
              smooth: 'Gaussian', 'flat'
              smooth_coefficient: std for Gaussian, peak height for flat
        '''
        super().__init__()
        self.norm_flag = norm
        if norm == 'softplus':
            self.norm = nn.Softplus()
        elif norm == 'relu':
            self.norm = nn.ReLU()
        elif norm == 'square':
            self.norm = Square()
        elif norm == 'abs':
            self.norm = ABS()
        elif norm == 'logistic':
            self.norm = Logistic()
        elif norm == 'softmax':
            self.norm = nn.Softmax(dim=1)
        else:
            raise Exception(f"Sorry, there is no such option as {norm} for normalization")
        
        self.smooth = smooth
        if self.smooth == 'Gaussian':
            self.std = smooth_coefficient
        elif self.smooth == 'flat':
            # value of the peak
            self.peak = smooth_coefficient 
        else:
            raise Exception(f"Sorry, there is no such option as {smooth} for smoothing")
        
    def forward(self, likelihood, gnd_idx, flag=False):
        '''
        for a batch, 
        param likelihood: FC layer output before normalization
              gnd_idx: ground truth bin idx 
              flag: add entropy term of likelihood
        '''
        self.batch_size, self.num_bins = likelihood.shape
        
        # normalize prediction
        if self.norm_flag == 'softmax':
            likelihood = self.norm(likelihood)
        else:
            likelihood = ((1e-7/self.num_bins+self.norm(likelihood)).T / (1e-7 + torch.sum(self.norm(likelihood), dim=1))).T
        cum_likelihood = torch.cumsum(likelihood, dim=1)
        
        # smooth gnd
        if self.smooth == 'Gaussian':
            gnd_dist = self.smooth_gnd_Guassian(gnd_idx, std=self.std, num_bins=self.num_bins, flat=False)
        else:
            gnd_dist = self.smooth_gnd(gnd_idx, self.num_bins, bin_val=self.peak)
        cum_gnd = torch.cumsum(gnd_dist, dim=1)
        
        # get distance by closed-form solution of W^1
        d = torch.abs(cum_likelihood-cum_gnd).sum() / self.batch_size 
        
        if flag:
            entropy = torch.sum(Categorical(probs=likelihood).entropy()) / 10 
            return d, entropy 
        if torch.isnan(d):
            print('nan')
        return d
    
    def smooth_gnd(self, gnd_idxs, num_bins:int, bin_val:float=0.9):
        '''
        label smoothing for a batch
        '''
        dist = torch.ones((self.batch_size, num_bins)) * (1-bin_val) / (num_bins-1)
        row = np.arange(0, self.batch_size)
        dist[row, gnd_idxs] = bin_val
        return dist.to(device=config.device, non_blocking=True)

    def smooth_gnd_Guassian(self, gnd_idx, std:float, num_bins:int, flat:bool=False):
        '''
        turn one-hot gnd distribution into discrete Gussian distribution from [0, 99]
        gnd_idx: a batch vector  
        '''
        a = torch.arange(0, num_bins).unsqueeze(dim=1)
        b = torch.cat([a]*self.batch_size, axis=1).T
        std = torch.full(size=(self.batch_size,1), fill_value=float(std))
        dist = torch.from_numpy(norm.pdf(b, gnd_idx.unsqueeze(dim=1).cpu().numpy(), std)) 
        if flat:
            dist += torch.ones(dist.shape) * 5*1e-4
        normalized_dist = (dist.T / torch.sum(dist, dim=1)).T # sum to 1
        return normalized_dist.to(device=config.device, non_blocking=True)
    

class EMDRenorm(nn.Module):
    '''
    closed form solution of Wasserstein distance in the case of 1D distribution
    Only the peaks' values that is above a threshold will be used for loss calculation. 
    Thus, both the prediction distribution and the Gaussian-smoothed gnd will subtract the threshold and re-normalized so they sum to one.
    And then EMD will be calculated between two renormalized distribution as the loss 
    '''
    def __init__(self, thresh) -> None:
        '''
        param thresh: threshold 
        '''
        super().__init__()
        self.norm = nn.Softplus()
        self.thresh = torch.tensor(thresh)
        self.thresh.to(device=config.device, non_blocking=True)
        self.std = 4
        
    def forward(self, likelihood, gnd_idx):
        '''
        for a batch, 
        set the prediction to zero if it cause division by zero, gnd is still renormalized
        param likelihood: FC layer output before normalization
              gnd_idx: ground truth bin idx 
              flag: add entropy term of likelihood
        '''
        self.batch_size, self.num_bins = likelihood.shape
        self.thresh = self.thresh*torch.ones(self.num_bins).to(device=config.device, non_blocking=True)
        
        likelihood = ((1e-7/self.num_bins+self.norm(likelihood)).T / (1e-7 + torch.sum(self.norm(likelihood), dim=1))).T
        re_likelihood, nan_idx = self.renormalize(likelihood)
        if nan_idx.shape[0] != 0:
            for idx in nan_idx:
                re_likelihood[idx, :] = torch.zeros(1, self.num_bins).to(device=config.device, non_blocking=True) 
        
        gnd_dist = self.smooth_gnd_Guassian(gnd_idx, std=self.std, num_bins=self.num_bins)       
        gnd_dist, _ = self.renormalize(gnd_dist)
            
        cum_likelihood = torch.cumsum(re_likelihood, dim=1)    
        cum_gnd = torch.cumsum(gnd_dist, dim=1)
        
        # get distance by closed-form solution of W^1
        d = torch.abs(cum_likelihood-cum_gnd).sum() / self.batch_size  
        return d
    
    def renormalize(self, dist):
        '''
        given a distribution, minus the threshold, only calculate the peak
        '''
        dist = torch.nn.functional.relu(dist-self.thresh)
        # check if there is one or more samples with all bins below threshhold
        sample_idx = (torch.sum(dist, dim=1) == 0).nonzero()
        if sample_idx.shape[0] != 1:
            sample_idx = sample_idx.squeeze()
        normalized_dist = (dist.T / torch.sum(dist, dim=1)).T
        return normalized_dist, sample_idx

    def smooth_gnd_Guassian(self, gnd_idx, std:float, num_bins:int):
        '''
        turn one-hot gnd distribution into discrete Gussian distribution from [0, 99]
        or Gaussian + uniform distribution
        gnd_idx: a batch vector  
        '''
        a = torch.arange(0, num_bins).unsqueeze(dim=1)
        b = torch.cat([a]*self.batch_size, axis=1).T
        std = torch.full(size=(self.batch_size,1), fill_value=float(std))
        dist = torch.from_numpy(norm.pdf(b, gnd_idx.unsqueeze(dim=1).cpu().numpy(), std)) 
        normalized_dist = (dist.T / torch.sum(dist, dim=1)).T # sum to 1
        return normalized_dist.to(device=config.device, non_blocking=True)

    
class EMDRenormTwo(EMDRenorm):
    '''
    loss for two grount truth
    '''
    def __init__(self, thresh, flag:True) -> None:
        '''
        dataloader for mixed-length GND annotation
        param: flag: True for two lines in annotation, false for only the first line        
        '''
        super().__init__(thresh)
        self.flag = flag
        
    def forward(self, likelihood, gnd_idx_1, gnd_idx_2):
        '''
        for a batch, 
        set the prediction to zero if it cause division by zero, gnd is still renormalized
        param likelihood: FC layer output before normalization
              gnd_idx: ground truth bin idx 
              flag: add entropy term of likelihood
        '''
        self.batch_size, self.num_bins = likelihood.shape
        self.thresh = self.thresh*torch.ones(self.num_bins).to(device=config.device, non_blocking=True)
        
        likelihood = ((1e-7/self.num_bins+self.norm(likelihood)).T / (1e-7 + torch.sum(self.norm(likelihood), dim=1))).T
        re_likelihood, nan_idx = self.renormalize(likelihood)
        if nan_idx.shape[0] != 0:
            for idx in nan_idx:
                re_likelihood[idx, :] = torch.zeros(1, self.num_bins).to(device=config.device, non_blocking=True) 
        cum_likelihood = torch.cumsum(re_likelihood, dim=1)
        
        gnd_dist = self.smooth_2gnd_Guassian(gnd_idx_1, gnd_idx_2, std=self.std, num_bins=self.num_bins)        
        cum_gnd = torch.cumsum(gnd_dist, dim=1)

        # get distance by closed-form solution of W^1
        d = torch.abs(cum_likelihood-cum_gnd).sum() / self.batch_size 
        return d
    
    def smooth_2gnd_Guassian(self, gnd_idx_1, gnd_idx_2, std: float, num_bins: int):
        '''
        turn one-hot gnd distribution into discrete Gussian distribution from [0, 99]
        or Gaussian + uniform distribution
        gnd_idx: a batch vector  
        '''
        a = torch.arange(0, num_bins).unsqueeze(dim=1)
        b = torch.cat([a]*self.batch_size, axis=1).T
        std1 = torch.full(size=(self.batch_size,1), fill_value=float(std))
        dist = torch.from_numpy(norm.pdf(b, gnd_idx_1.unsqueeze(dim=1).cpu().numpy(), std1)) 
        
        if self.flag: # second peak in GND
            non_inf_mask = torch.where(gnd_idx_2!=torch.inf)
            num_non_inf =  int(torch.where(gnd_idx_2!=torch.inf, 1.0, 0.0).sum().item())  # when there is only one image in the batch, it doesn't work
            c = torch.cat([a]*num_non_inf, axis=1).T
            std2 = torch.full(size=(num_non_inf, 1), fill_value=float(std)) 
            dist[non_inf_mask] += torch.from_numpy(norm.pdf(c, gnd_idx_2[non_inf_mask].unsqueeze(dim=1).cpu().numpy(), std2))        
        normalized_dist = (dist.T / torch.sum(dist, dim=1)).T # sum to 1
        return normalized_dist.to(device=config.device, non_blocking=True)


class EntropyEMDRenormTwo(EMDRenormTwo):
    def forward(self, likelihood, gnd_idx_1, gnd_idx_2):
        '''
        for a batch, 
        set the prediction to zero if it causes division by zero, gnd is still renormalized
        param likelihood: FC layer output before normalization
              gnd_idx: ground truth bin idx 
              flag: add entropy term of likelihood
        '''
        self.batch_size, self.num_bins = likelihood.shape
        self.thresh = self.thresh*torch.ones(self.num_bins).to(device=config.device, non_blocking=True)
        
        likelihood = ((1e-7/self.num_bins+self.norm(likelihood)).T / (1e-7 + torch.sum(self.norm(likelihood), dim=1))).T
        re_likelihood, nan_idx = self.renormalize(likelihood)
        if nan_idx.shape[0] != 0:
            for idx in nan_idx:
                re_likelihood[idx, :] = torch.zeros(1, self.num_bins).to(device=config.device, non_blocking=True) 
        cum_likelihood = torch.cumsum(re_likelihood, dim=1)
        
        gnd_dist = self.smooth_2gnd_Guassian(gnd_idx_1, gnd_idx_2, std=self.std, num_bins=self.num_bins)        
        cum_gnd = torch.cumsum(gnd_dist, dim=1)

        # get distance by closed-form solution of W^1
        d = torch.abs(cum_likelihood-cum_gnd).sum() / self.batch_size 
        entropy = torch.sum(Categorical(probs=likelihood).entropy()) / 10
        return d, entropy

class CRPS(EMDRenorm):
    def __init__(self, thresh, bin_center, flag:True) -> None:
        super(CRPS, self).__init__(thresh)
        self.bin_center = torch.tensor(bin_center)
        self.bin_center.to(device=config.device, non_blocking=True)
        self.flag = flag
        
    def forward(self, likelihood, gnd_idx_1, gnd_idx_2, idx=True):
        '''
        for a batch, 
        set the prediction to zero if it cause division by zero, gnd is still renormalized
        param likelihood: FC layer output before normalization
              gnd_idx: ground truth bin idx 
              smooth: False for caluclating CRPS to crisp GND; True for Gussian smoothing
              idx: True for index space; False for parameter space
        '''
        self.batch_size, self.num_bins = likelihood.shape
        self.thresh = self.thresh*torch.ones(self.num_bins).to(device=config.device, non_blocking=True)
        
        likelihood = ((1e-7/self.num_bins+self.norm(likelihood)).T / (1e-7 + torch.sum(self.norm(likelihood), dim=1))).T
        re_likelihood, nan_idx = self.renormalize(likelihood)
        if nan_idx.shape[0] != 0:
            for idx in nan_idx:
                re_likelihood[idx, :] = torch.zeros(1, self.num_bins).to(device=config.device, non_blocking=True) 
        
        gnd_dist = self.get_gnd_onehot(gnd_idx_1, gnd_idx_2)
            
        if idx:
            cum_likelihood = torch.cumsum(re_likelihood, dim=1)    
            cum_gnd = torch.cumsum(gnd_dist, dim=1)
        else: # parameter space, weighted by bin centrum THIS IS NOT TRUE!!
            cum_likelihood = torch.cumsum(re_likelihood*self.bin_center, dim=1)    
            cum_gnd = torch.cumsum(gnd_dist*self.bin_center, dim=1)

        d = torch.square(cum_likelihood-cum_gnd).sum() / self.batch_size 
        return d
    
    def get_gnd_onehot(self, gnd_idx_1, gnd_idx_2):
        gnd_dist = torch.zeros(size=(self.batch_size, self.num_bins))
        gnd_dist[:, gnd_idx_1] = 0.5
        if self.flag: 
            if gnd_idx_2!= torch.inf:
                gnd_dist[:, gnd_idx_2] = 0.5
        return gnd_dist.to(device=config.device, non_blocking=True)
    
