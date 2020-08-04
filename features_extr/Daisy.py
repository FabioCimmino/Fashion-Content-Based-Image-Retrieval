from __future__ import print_function

from skimage.feature import daisy
from skimage import color
from skimage import data
from scipy import spatial
from six.moves import cPickle
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import os
import cv2


n_slice    = 2
n_orient   = 8
step       = 15
radius     = 8
rings      = 3
histograms = 8
h_type     = 'global'
d_type     = 'd1'
depth      = 3
R = (rings * histograms + 1) * n_orient

def histogram(input, type=h_type, n_slice=n_slice, normalize=True):
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = cv2.imread(input)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape

    P = math.ceil((height - radius*2) / step) 
    Q = math.ceil((width - radius*2) / step)
    assert P > 0 and Q > 0, "input image size need to pass this check"

    hist = daisy_launcher(img)
  
    if normalize:
      hist /= np.sum(hist)

    return hist.flatten()




def daisy_launcher(img, normalize=True):
    image = color.rgb2gray(img)
    descs = daisy(image, rings=rings, histograms=histograms, orientations=n_orient)
    #descs, descs_img = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=n_orient, visualize=True)
    #descs_num = descs.shape[0] * descs.shape[1]
    descs = descs.reshape(-1, R)  # shape=(N, R)
    hist  = np.mean(descs, axis=0)  # shape=(R,)
  
    '''
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(descs_img)
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.show()
    '''
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
    return hist




def daisy_extraction (classe_pred, img_path, dir_dataset, dirImgOut):

  query = histogram(img_path)
  image_list = []
  img_dir = dir_dataset + '/' + classe_pred + '/*.jpg'
  dist_list = []
  name_list = []
  for filename in glob.glob(img_dir):
    hist = histogram(filename)
    '''
    image_list.append({
                        distance(query, hist, 'd2'),
                        filename
                      })
    '''
    dist_list.append(distance(query, hist, 'd2'))
    name_list.append(filename)
  
  image_list = list( zip(name_list, dist_list))

  image_list.sort(key=takeSecond)
  image_list = image_list[:10]
  
  results = []
  for img in image_list:
    results.append(takeFirst(img))

  print (results)
  #fig=plt.figure(figsize=(8, 8))
  columns = 5
  rows = 2
  plt.axis('off')
  f, axarr = plt.subplots(2, 5)
  idx=0
  f.suptitle('DAISY Results')
  for i in range(rows):
      for j in range (columns):
          if isinstance(results[idx], np.ndarray):  # examinate input type
            img = results[idx].copy()
          else:
            img = cv2.imread(results[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          axarr[i,j].imshow(img)
          axarr[i,j].axis('off')
          axarr[i,j].set_title(idx+1)
          idx+=1
  #plt.show()

  plt.savefig(dirImgOut+'/Daisy_result.jpg')






def takeFirst(elem):
    return elem[0]

def takeSecond(elem):
    return elem[1]


def distance(v1, v2, d_type):
  assert v1.shape == v2.shape, "shape of two vectors need to be same!"

  if d_type == 'd1':
    return np.sum(np.absolute(v1 - v2))
  elif d_type == 'd2':
    return np.sum((v1 - v2) ** 2)
  elif d_type == 'd2-norm':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd3':
    pass
  elif d_type == 'd4':
    pass
  elif d_type == 'd5':
    pass
  elif d_type == 'd6':
    pass
  elif d_type == 'd7':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd8':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'cosine':
    return spatial.distance.cosine(v1, v2)
  elif d_type == 'square':
    return np.sum((v1 - v2) ** 2)