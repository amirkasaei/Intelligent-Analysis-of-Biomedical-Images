import pandas as pd
import glob
import cv2
import os
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def load_mri_df(mri_scans_path):
    def pos_neg_diagnosis(mask_path):
        value = np.max(cv2.imread(mask_path))
        if value > 0 :
            return 1
        else:
            return 0
            
    data_map = []
    for sub_dir_path in glob.glob(mri_scans_path+"*"):
        try:
            dir_name = sub_dir_path.split('/')[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + '/' + filename
                data_map.extend([dir_name, image_path])
        except Exception as e:
            print(e)
    
    
    df = pd.DataFrame(
        {
            "patient_id" : data_map[::2],
            "path" : data_map[1::2]
        }
    )
    
    df_imgs = df[~df['path'].str.contains("mask")]
    df_masks = df[df['path'].str.contains("mask")]
    
    imgs = df_imgs["path"].values
    masks = df_masks["path"].values
    
    mri_df = pd.DataFrame(
        {
            "patient_id": df_imgs.patient_id.values,
            "image_path": imgs,
            "mask_path": masks,
        }
    )

    mri_df['has_cancer'] = mri_df['mask_path'].apply(lambda x: pos_neg_diagnosis(x))
    return mri_df