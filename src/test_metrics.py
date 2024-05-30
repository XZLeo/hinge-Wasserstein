import numpy as np
from sklearn.metrics import auc
import torch
from src.functionals import bin_normal2leftright, normal2leftright
import matplotlib.pyplot as plt


def get_horizon_error(gnd_left, gnd_right, pred_left, pred_right, input_size=224):
    '''
    the maximum distance from the detection to the ground truth in the cropped resized image space,
    param: gnd_left: y coordinate on the cropped resized image          
    '''
    return max(abs(gnd_left-pred_left), abs(gnd_right-pred_right))/input_size


def get_batch_horizon_error(outputs, targets, bin_edges, batch_size:int)->list:
    '''ambiguity'''
    horizon_errors = []
    for idx in range(batch_size):        
        output_theta = outputs.theta[idx, :] # batch_size * categories
        output_rho = outputs.rho[idx, :]
       
        target_theta = targets[0][idx].cpu().numpy()
        target_rho = targets[1][idx].cpu().numpy()
        # (theta, rho) ==> (left, right)
        output_left, output_right, _, _ = bin_normal2leftright(output_theta.argmax(), output_rho.argmax(), bin_edges)
        target_left, target_right = normal2leftright(target_theta, target_rho)
        horizon_error = get_horizon_error(target_left, target_right, output_left, output_right)
        horizon_errors.append(horizon_error)
    return horizon_errors


def get_AUC(horrizon_errors:list):
    num_samples = len(horrizon_errors)
    fractions = []
    horrizon_errors = np.array(horrizon_errors)
    thresholds = np.arange(0, 0.25, 0.01) 
    for thresh in thresholds:
        fractions.append(np.where(horrizon_errors<thresh)[0].size/num_samples)
    fractions = np.array(fractions)
    # get AUC
    area_under_curve = auc(thresholds, fractions)        
    return area_under_curve


class Metrics:
    def __init__(self) -> None:
        self.joint = {"AUC":[], "AUSE":[]}
        self.theta = {"RMSE":[], "CRPS":[], "NLL":[], "AUSE":[], "ECE":[]}
        self.rho = {"RMSE":[], "CRPS":[], "NLL":[], "AUSE":[], "ECE":[]}
                
    def get_mean_std(self) -> None:
        result = ""
        for metric, values in vars(self).items():
            for key, tables in values.items():
                result += "{} {} mean {:.4} std {:.4}\n".format(
                    metric, key, np.mean(tables), np.std(tables))
        return result
    
    def update(self, metrics, flag):
        """
        add metric from a new model into list
        Args:
            metrics (list): length is 2/4, order: rmse, crps, nll, ause/auc, ause
            flag (str): choosing attirbute
        """
        if flag == 'theta':
            for idx, key in enumerate(self.theta.keys()): 
                self.theta[key].append(metrics[idx])
        elif flag == 'rho':
            for idx, key in enumerate(self.rho.keys()): 
                self.rho[key].append(metrics[idx])
        else:
            for idx, key in enumerate(self.joint.keys()): 
                self.joint[key].append(metrics[idx])
        return


def get_brier_ece(cdfs, vis=False, savefig_filename=None):
    cdfs, _ = cdfs.sort()
    intervals = torch.arange(0.1, 1.1, step=0.1)
    ecdfs = get_ecdf(cdfs)
    ece = torch.sum(torch.square(intervals-ecdfs))
    if vis:
        fig, axis = plt.subplots(1, 1)
        axis.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
        axis.plot(intervals, ecdfs, "-o", alpha=0.7)
        axis.set_xlabel('Predicted')
        axis.set_ylabel('Empirical')
        axis.xaxis.set_ticks([0.0, 1.0])
        axis.yaxis.set_ticks([0.0, 1.0])
        axis.ticklabel_format(useMathText=True)
        fig.suptitle('Predicted CDF vs Empirical CDF')
        fig.tight_layout()
        fig.legend()
        fig.subplots_adjust(right=0.825)
        if savefig_filename is not None:
            fig.set_size_inches(8, 4.5)
            plt.savefig(savefig_filename)
        plt.show()
    return ece, ecdfs


def get_ecdf(predicted_cdf):
    '''Empirical CDF.

    Gets the empirical cdf, also represented as $\hat{P}[H(x_t)(y_t)]$ in the paper.
    Counts how many points in the dataset have a pcdf <= to the pcdf of a point for all points in the dataset.

    Parameters
    ----------
    predicted_cdf : np.array
        Predicted cdf. Can be generated by calling self.pcdf for posterior predictive samples at a particular quantile.

    Returns
    -------
    ecdf_ : np.array
        The empirical cdf

    '''
    intervals = torch.arange(0.1, 1.1, step=0.1)
    empirical_cdf = torch.zeros(10)
    T = len(predicted_cdf)
    for i, p in enumerate(intervals):
        empirical_cdf[i] = torch.sum(predicted_cdf <= p)/T
    return empirical_cdf 
   
