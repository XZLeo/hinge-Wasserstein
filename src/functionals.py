import numpy as np
from math import sqrt
from collections import namedtuple


SubwindowCenter = namedtuple("Subwindow", ["x", "y", "dim"]) 


def val2bin(theta, rho, bin_edges:dict):
    '''
    For the classification approach, transfer (theta, rho) parameterization to bin_id
    param: bin_edges: contains two sorted sequences from smallest to largest
    return theta_bin_id: in [0, 99]
           rho_bin_id: in [0, 99]
    '''
    slope_edges = bin_edges['slope_bin_edges']
    offset_edges = bin_edges['offset_bin_edges']

    theta_id = np.where(slope_edges>theta)[0][0]-1
    rho_id =  np.where(offset_edges>rho)[0][0]-1   
      
    return theta_id, rho_id 

def val2bin_linint(theta, rho, bin_edges:dict):
    '''
    Compute a linearly interpolated "index" for (theta, rho) 
    param: bin_edges: contains two sorted sequences from smallest to largest
    return theta_bin_id: in [0, 99]
           rho_bin_id: in [0, 99]
    '''
    slope_edges = bin_edges['slope_bin_edges']
    slope_midpoints = (slope_edges[:-1]+slope_edges[1:])/2.0
    
    offset_edges = bin_edges['offset_bin_edges']
    offset_midpoints = (offset_edges[:-1]+offset_edges[1:])/2.0
    
    theta_il = np.where(slope_midpoints>theta)[0][0]-1 # Lower edge
    d1 = theta-slope_midpoints[theta_il]
    d2 = slope_midpoints[theta_il+1]-theta
    lbd = d2/(d1+d2)
    theta_i = theta_il*lbd + (theta_il+1)*(1.0-lbd)
    
    rho_il =  np.where(offset_midpoints>rho)[0][0]-1 # Lower edge
    d1 = rho-offset_midpoints[rho_il]
    d2 = offset_midpoints[rho_il+1]-rho
    lbd = d2/(d1+d2)
    
    rho_i = rho_il*lbd + (rho_il+1)*(1.0-lbd)
      
    return theta_i, rho_i

def bin2val(bin_id, bin_edges:np.ndarray):
  '''
  for classificaiton, transfer class label to value
  param: bin_id: int for classification prediction
         bin_edges: float, the width of a bin
  '''
  assert 0 <= bin_id < bin_edges.size-1, 'impossible bin_id'

  # handle infinite bins, choose left/right edge as appropriate
  if bin_id == 0 and bin_edges[0] == -np.inf:
    val = bin_edges[1]
  elif bin_id == bin_edges.size-2 and bin_edges[-1] == np.inf:
    val = bin_edges[-2]
  else:
    val = (bin_edges[bin_id] + bin_edges[bin_id+1]) / 2
    # get the mean of two bin edge
  return val


def leftright2normal(left_y:float, left_x:float, right_y:float, right_x:float, crop_size):
    '''
    parameterization (l,r) ==> (theta, rho)
    rho = x*cos(theta) + y*sin(theta)
    theta in [-pi/2, pi/2]
    param: left_y: annotation left edge's x coordinate
           right_y: right edge's x coordinate
           crop_size: size of square subwindow
    '''
    width = abs(left_x-right_x)
    r = sqrt((left_y-right_y)*(left_y-right_y)/width/width+1)   # 1/cos(theta)
    b = (left_y+right_y) / 2
    rho = b / r / crop_size 
    # in units of image heights
    # relative value doesn't change with resize
    theta = np.arctan((right_y-left_y)/(right_x-left_x))  
    return theta, rho

def subwindow_leftright2normal(left_y:float, left_x:float, right_y:float,
                     right_x:float, sub_y:float, sub_x:float, crop_size):
    '''
    parameterization (l,r) ==> (theta, rho)
    rho = x*cos(theta) + y*sin(theta)
    theta in [-pi/2, pi/2]
    param: left_x, left_y: annotation left edge's x coordinate
           right_x, right_y: right edge's x coordinate
           sub_x,sub_y: central of square subwindow
           all in uncroped center coordinate
           crop_size: the size of subwindow
    '''
    width = abs(left_x-right_x)
    r = sqrt((left_y-right_y)*(left_y-right_y)/width/width+1)   
    a = abs((sub_x-left_x)/(right_x-left_x))
    b = abs((right_x-sub_x)/(right_x-left_x))
    sub_intercept = b*left_y + a*right_y
    rho = (sub_intercept-sub_y) / r / crop_size 
    # in units of image heights
    # relative value doesn't change with resize
    theta = np.arctan((right_y-left_y)/(right_x-left_x))  
    return theta, rho

def normal2leftright(slope:float, offset:float, caffe_sz=224):
  '''
  transfer (theta, rho) parameterization to (left_y, right_y) in the resized image space 
  param: caffe_sz: 224
         slope_dist: the output list before softmax
         offset_dist: the output list before softmax, normalized by image height
         bin_edges: the stored list for both theta and rho
  '''
  # (slope, offset) to (left, right) in the cropped resized img
  slope = slope #- np.pi/2
  offset = offset * caffe_sz
  c = offset / np.cos(np.abs(slope))
  left = -np.tan(slope)*caffe_sz/2 + c
  right = np.tan(slope)*caffe_sz/2 + c
  return left, right


def bin_normal2leftright(slope_bin, offset_bin, bin_edges, caffe_sz=224):
  '''
  transfer (theta, rho) parameterization to (left_y, right_y) in the resized image space 
  param: caffe_sz: 224
         slope_dist: the output list before softmax
         offset_dist: the output list before softmax
         bin_edges: the stored list for both theta and rho
  '''
  # compute (slope, offset)
  slope = bin2val(slope_bin, bin_edges['slope_bin_edges']) #- np.pi/2
  offset = bin2val(offset_bin, bin_edges['offset_bin_edges'])
  # (slope, offset) to (left, right) in the cropped resized img
  offset_test = offset * caffe_sz  # on y axis
  c = offset_test / np.cos(np.abs(slope))
  left = -np.tan(slope)*caffe_sz/2 + c
  right = np.tan(slope)*caffe_sz/2 + c
  return left, right, slope, offset


def extrap_horizon(left, right, width):
  
  hl_homo = np.cross(np.append(left, 1), np.append(right, 1))
  hl_left_homo = np.cross(hl_homo, [-1, 0, -width/2])
  hl_left = hl_left_homo[0:2]/hl_left_homo[-1]
  hl_right_homo = np.cross(hl_homo, [-1, 0, width/2])
  hl_right = hl_right_homo[0:2]/hl_right_homo[-1]
  
  return hl_left, hl_right


def compute_horizon(slope_dist, offset_dist, caffe_sz, sz, crop_info, bin_edges):
  '''
  param: caffe_sz: 224
  '''
  # setup
  crop_sz, x_inds, y_inds = crop_info

  # get maximum bin 
  slope_bin = np.argmax(slope_dist)
  offset_bin = np.argmax(offset_dist)
  
  # compute (slope, offset)
  slope = bin2val(slope_bin, bin_edges['slope_bin_edges']) 
  offset = bin2val(offset_bin, bin_edges['offset_bin_edges'])
  
  # (slope, offset) to (left, right) in the cropped resized img
  offset = offset * caffe_sz[0]
  c = offset / np.cos(np.abs(slope))
  caffe_left = -np.tan(slope)*caffe_sz[1]/2 + c
  caffe_right = np.tan(slope)*caffe_sz[1]/2 + c

  # scale back to cropped image
  c_left = caffe_left * (crop_sz[0] / caffe_sz[0])
  c_right = caffe_right * (crop_sz[0] / caffe_sz[0])

  # scale back to original image
  center = [(sz[1]+1)/2, (sz[0]+1)/2]
  crop_center = [np.dot(x_inds,[.5, .5])-center[0], center[1]-np.dot(y_inds,[.5, .5])]
  left_tmp = np.asarray([-crop_sz[1]/2, c_left]) + crop_center 
  right_tmp = np.asarray([crop_sz[1]/2, c_right]) + crop_center 
  left, right = extrap_horizon(left_tmp, right_tmp, sz[1])
  return left, right


