# Imports
import cv2
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from tqdm.notebook import tqdm
import h5py

def preprocess(data_root_path  , img_dim  , outpath , num_samples = None , channels=1 , mode="train" , normalize=True):
  files = glob.glob(f"{DATA_ROOT}/Mass*/*/")
  if channels == 1:
    read = cv2.IMREAD_GRAYSCALE
  elif channels ==3:
    read = cv2.IMREAD_COLOR

  if normalize == True:
    factor = 255.
  else:
    factor = 1.

  if num_samples == None:
    limit = len(files)
  else:
    limit = num_samples

  for i in tqdm(range(limit)):
    if i == 0:
      all_imgs = cv2.resize(cv2.imread(glob.glob(f"{files[105]}/*/*.png")[0] , read) , (img_dim))[np.newaxis , : , :] / factor
    else: 
      new_img = cv2.resize(cv2.imread(glob.glob(f"{files[105]}/*/*.png")[0] ,  read) , (img_dim))[np.newaxis , : , :] / factor
      all_imgs = np.vstack((all_imgs , new_img) ) 

  os.makedirs(os.path.dirname(outpath), exist_ok=True)

  h5f = h5py.File(outpath, 'w')
  h5f.create_dataset(mode, data=all_imgs)

  h5f.close()
