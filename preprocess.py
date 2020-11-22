# Imports
import cv2
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
from sklearn import preprocessing


def get_clinical_helper(i , g_t_dataset , read, files , data_root_path, img_dim):
   index_row = g_t_dataset.loc[g_t_dataset['image file path'] == files]
   breast_density = np.array(index_row['breast_density'])[np.newaxis , : ]
   side =  np.array(index_row['left or right breast'] == "RIGHT")[np.newaxis , : ]
   mass_shape =  np.array(index_row['mass shape'], dtype=str)[np.newaxis , : ]
   mass_margins =  np.array(index_row['mass margins'], dtype=str)[np.newaxis , : ]
   pathology =  np.array(index_row['pathology'], dtype=str)[np.newaxis , : ]
   subtlety = np.array(index_row['subtlety'] , dtype=str)[np.newaxis , : ]
   path = f"{data_root_path}/{index_row['ROI mask file path'][  g_t_dataset.index[i]]}"
   roi_paths =  glob.glob(f"{path[:-1]}*.png")

   roi = cv2.imread(roi_paths[0] , read)
   roi_mask = cv2.imread(roi_paths[1], read)

   roi = cv2.resize(roi , img_dim)[np.newaxis , : , :] / 255.
   roi_mask = cv2.resize(roi_mask , img_dim)[np.newaxis , : , :] / 255.
   return breast_density , side , mass_shape , mass_margins , pathology , roi , roi_mask



def preprocess(data_root_path  , img_dim  , outpath , abnormality , label_file_path , channels=1 , mode="train" , normalize=True, num_samples=None, view="CC"):

  g_t_dataset = pd.read_csv(label_file_path)
  g_t_dataset = g_t_dataset.loc[g_t_dataset['image view'] == view]
  g_t_dataset = g_t_dataset.loc[g_t_dataset['abnormality type'] == abnormality]
  if channels == 1:
    read = cv2.IMREAD_GRAYSCALE
  elif channels ==3:
    read = cv2.IMREAD_COLOR

  if normalize == True:
    factor = 255.
  else:
    factor = 1.

  if num_samples == None:
    limit = len(g_t_dataset)
  else:
    limit = num_samples

  for i in tqdm(range(limit)):
    if i == 0:
      all_imgs = cv2.resize(cv2.imread(glob.glob(f"{data_root_path}/{g_t_dataset['image file path'][  g_t_dataset.index[i]]}*.png")[0] , read) , (img_dim))[np.newaxis , : , :] / factor
      all_breast_density , all_side , all_mass_shape , all_mass_margins , all_pathology , all_roi , all_roi_mask = get_clinical_helper(i , g_t_dataset , read , g_t_dataset['image file path'][i] , data_root_path, img_dim)

    else:
      all_imgs = np.vstack((all_imgs , cv2.resize(cv2.imread(glob.glob(f"{data_root_path}/{g_t_dataset['image file path'][ g_t_dataset.index[i]]}*.png")[0] , read) , (img_dim))[np.newaxis , : , :] / factor) )
      clinical_tuple =  get_clinical_helper(i , g_t_dataset , read , g_t_dataset['image file path'][  g_t_dataset.index[i]] , data_root_path , img_dim)
      all_breast_density = np.vstack((all_breast_density , clinical_tuple[0]))
      all_side = np.vstack((all_side , clinical_tuple[1]))
      all_mass_shape =  np.vstack((all_mass_shape , clinical_tuple[2]))
      all_mass_margins = np.vstack((all_mass_margins , clinical_tuple[3]))
      all_pathology = np.vstack((all_pathology , clinical_tuple[4]))
      all_roi = np.vstack((all_roi, clinical_tuple[5]))
      all_roi_mask = np.vstack((all_roi_mask , clinical_tuple[6]))



  print(all_mass_shape)
  os.makedirs(os.path.dirname(outpath), exist_ok=True)

  h5f = h5py.File(outpath, 'w')
  h5f.create_dataset(f"{mode}-raw-imgs", data=all_imgs)
  h5f.create_dataset(f"{mode}-breast_density", data=all_breast_density)
  h5f.create_dataset(f"{mode}-side", data=all_side)
  le = preprocessing.LabelEncoder()
  h5f.create_dataset(f"{mode}-mass_shape", data=le.fit_transform(all_mass_shape))
  le = preprocessing.LabelEncoder()
  h5f.create_dataset(f"{mode}-mass_margins", data=le.fit_transform(all_mass_margins))
  le = preprocessing.LabelEncoder()
  h5f.create_dataset(f"{mode}-pathology", data=le.fit_transform(all_pathology))
  h5f.create_dataset(f"{mode}-roi", data=all_roi)
  le = preprocessing.LabelEncoder()
  h5f.create_dataset(f"{mode}-roi_mask", data=all_roi_mask)


  h5f.close()
