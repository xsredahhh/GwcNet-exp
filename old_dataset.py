import torch
import os, os.path
from torch.utils.data import Dataset
import pandas as pd
import cv2
import tifffile
import albumentations as albu
from albumentations.pytorch.functional import img_to_tensor
import numpy as np


class MyDataset(Dataset):
    def __init__(self, file_path, transform=None, downsampling=2,  phase='train', store_img_infor_root=None, store_matrix_root=None, store_baseline_root=None, store_focal_root=None,  interval=1):
        self.file_path = file_path
        self.transform = transform
        self.downsampling = downsampling
        self.store_img_infor_root = store_img_infor_root
        self.store_matrix_root = store_matrix_root
        self.store_baseline_root = store_baseline_root
        self.store_focal_root = store_focal_root
        self.phase = phase
        self.interval = interval

        if self.store_matrix_root is None:
            self.store_matrix_root = './{}_repro_matrixes.npz'.format(self.phase)
        if self.store_img_infor_root is None:
            self.store_img_infor_root = './{}_imgs_path.csv'.format(self.phase)
        if self.store_baseline_root is None:
            self.store_baseline_root = './{}_baselines.npy'.format(self.phase)
        if self.store_focal_root is None:
            self.store_focal_root = './{}_focal.npy'.format(self.phase)

        if os.path.exists(self.store_img_infor_root) and os.path.exists(self.store_matrix_root) and os.path.exists(self.store_baseline_root) and os.path.exists(self.store_focal_root):
            pass
        else:
            print("No Image information csv!!!")
            exit()

        self.rec_imgs_and_gts_names = pd.read_csv(self.store_img_infor_root)

        self.reprojecton_matrixs = np.load(self.store_matrix_root, allow_pickle=True)
        baseline = np.load(self.store_baseline_root)
        self.baseline = baseline.tolist()
        focal = np.load(self.store_focal_root)
        self.focal = focal.tolist()
        self.normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)

    def __len__(self):
        return self.rec_imgs_and_gts_names.shape[0]

    def __getitem__(self, idx):
        rec_left_path = self.rec_imgs_and_gts_names['Rectified_left_image_path'].values[idx]
        rec_right_path = self.rec_imgs_and_gts_names['Rectified_right_image_path'].values[idx]
        rec_left_gt_path = self.rec_imgs_and_gts_names['Rectified_left_gt_path'].values[idx]
        # rec_right_gt_path = self.rec_imgs_and_gts_names['Rectified_right_gt_path'].values[idx]
        # mask_left_path = self.rec_imgs_and_gts_names['Rectified_left_mask_path'].values[idx]
        # mask_right_path = self.rec_imgs_and_gts_names['Rectified_right_mask_path'].values[idx]

        reprojection_matrix_left = self.reprojecton_matrixs['arr_0'][idx]
        # reprojection_matrix_right = reprojection_matrix_left.copy()
        # reprojection_matrix_right[3][2] = - reprojection_matrix_right[3][2]
        # bb = self.baseline[idx]
        # ff = self.focal[idx]
        rec_left = cv2.imread(rec_left_path)
        rec_right = cv2.imread(rec_right_path)
        rec_left_gt = tifffile.imread(rec_left_gt_path)
        # rec_right_gt = tifffile.imread(rec_right_gt_path)


        ds_left = cv2.resize(rec_left, (0, 0), fx=1. / self.downsampling, fy=1. / self.downsampling)
        ds_right = cv2.resize(rec_right, (0, 0), fx=1. / self.downsampling, fy=1. / self.downsampling)
        del rec_left
        del rec_right
        # ds_left = ds_left.astype(np.float32)
        # ds_right = ds_right.astype(np.float32)
        rec_left_gt = rec_left_gt.astype(np.float32)
        # rec_right_gt = rec_right_gt.astype(np.float32)

        left_z = rec_left_gt[:, :, 2]
        # right_z = rec_right_gt[:, :, 2]
        med_left = np.median(left_z[left_z > 0])
        # med_right = np.median(right_z[right_z > 0])

        mask_left = (left_z > 5).astype(np.float32) * (left_z < 5 * med_left).astype(np.float32)
        # mask_right = (right_z > 5).astype(np.float32) * (right_z < 5 * med_right).astype(np.float32)
        mask_left = mask_left[..., np.newaxis]
        # mask_right = mask_right[..., np.newaxis]
        mask_left = np.repeat(mask_left, 3, axis=2)
        # mask_right = np.repeat(mask_right, 3, axis=2)
        # bb和ff转换成array，类型为float32
        reprojection_matrix_left = reprojection_matrix_left.astype(np.float32)
        # reprojection_matrix_right = reprojection_matrix_right.astype(np.float32)
        # bb = np.array(bb).astype(np.float32)
        # ff = np.array(ff).astype(np.float32)
        if self.phase == 'train':
            if self.transform is not None:
                ds_left = self.transform(image=ds_left)['image']
                ds_right = self.transform(image=ds_right)['image']
            ds_left = self.normalize(image=ds_left)['image']
            ds_right = self.normalize(image=ds_right)['image']
        else:
            ds_left = self.normalize(image=ds_left)['image']
            ds_right = self.normalize(image=ds_right)['image']

        return [img_to_tensor(ds_left), img_to_tensor(ds_right), img_to_tensor(rec_left_gt),\
               img_to_tensor(mask_left),  torch.from_numpy(reprojection_matrix_left)]
