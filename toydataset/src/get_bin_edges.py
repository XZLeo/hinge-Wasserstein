'''
For toy dataset, generate equally-spaced bin
'''
import numpy as np
from scipy.io import savemat

THETA_RANGE = [-np.pi/2, np.pi/2]
RHO_RANGE = [-50/224, 50/224]

bin_edges = {}
bin_edges['slope_bin_edges'] = np.linspace(THETA_RANGE[0], THETA_RANGE[1], num=101).reshape((101, 1))
bin_edges['offset_bin_edges'] = np.linspace(RHO_RANGE[0], RHO_RANGE[1], num=101).reshape((101, 1))
bin_edges['offset_bin_edges'][0] = -np.inf
bin_edges['offset_bin_edges'][-1] = np.inf

savemat('toydataset/toy_bins.mat', bin_edges)