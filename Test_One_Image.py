from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import torchsummary
import datetime
import tqdm
import cv2
import math
import albumentations as albu
from pathlib import Path
from tensorboardX import SummaryWriter
import old_dataset as MyD
from models import basic, stackhourglass, submodule
from processing import display
from albumentations.pytorch.functional import img_to_tensor
import gwc

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=96,
                    help='maxium disparity')
parser.add_argument('--model', default='GwcNet_GC',
                    help='select model')
parser.add_argument('--datapath', default='/home/greatbme/data_plus/Stereo Dataset',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='/home/greatbme/public/HX/GwcNet/depth_estimation_training_run_5_13_16_valid_2/checkpoint_2.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'GwcNet_GC':
    model = gwc.GwcNet_GC(int(args.maxdisp))
elif args.model == 'GwcNet_G':
    model = gwc.GwcNet_G(int(args.maxdisp))
else:
    print('no model')
if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.cuda()

if args.loadmodel is not None:
    print('load GwcNet')
    state_dict = torch.load(args.loadmodel)
    step = state_dict['step']
    epoch = state_dict['epoch']
    model.load_state_dict(state_dict['state_dict'])
    print('Restored model, epoch {}, step {}'.format(epoch, step))

# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def Test(ds_left, ds_right, reproj_left):
    model.eval()

    if args.cuda:
        ds_left, ds_right,  reproj_left = ds_left.cuda(), ds_right.cuda(), reproj_left.cuda()

    ds_left = ds_left.unsqueeze(0)
    ds_right = ds_right.unsqueeze(0)
    reproj_left = reproj_left.unsqueeze(0)

    with torch.no_grad():
        t1 = time.time()
        output3 = model(ds_left, ds_right)[0]
        b, h, w = output3.shape
        depth3 = submodule.reprojection()(output3, reproj_left)
        dur = time.time() - t1
        print("dur: ", dur)

    cp_z = depth3[:, 2, :, :]
    min_val = cp_z.min()
    max_val = cp_z.max()

    cp_z = ((cp_z - min_val) / (max_val - min_val)).data.cpu().numpy()
    heat_cp = cv2.applyColorMap(np.uint8(255 * np.moveaxis(cp_z, source=[0, 1, 2], destination=[2, 0, 1])),
                                cv2.COLORMAP_JET)
    depth3 = depth3.squeeze(0).permute([1,2,0]).data.cpu().numpy()
    return heat_cp, depth3


if __name__ == '__main__':
    for i in range(9,15):
        dir_left = '/home/greatbme/mydata/data_plus/Experiments/rectified/left{}.png'.format(i)
        dir_right = '/home/greatbme/mydata/data_plus/Experiments/rectified/right{}.png'.format(i)

        # repojection = np.load('/home/greatbme/mydata/data_plus/Stereo Dataset/dataset_4/keyframe_3/reprojection_matrix.npy')
        repojection = np.array(
            [(1, 0, 0, -892.133758), (0, 1, 0, -537.283478), (0, 0, 0, 1374.25961), (0, 0, 0.10151, 0)])

        print(repojection)
        left_img = cv2.imread(dir_left)
        right = cv2.imread(dir_right)
        left_img = cv2.resize(left_img, (0, 0), fx=1. / 2, fy=0.474)
        right = cv2.resize(right, (0, 0), fx=1. / 2, fy=0.474)
        print("left.shape: ", left_img.shape)
        normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
        left = normalize(image=left_img)['image']
        right = normalize(image=right)['image']

        heat_cp, depth = Test(img_to_tensor(left), img_to_tensor(right), torch.from_numpy(repojection))

        folder = Path('./cross_test_epoch{}'.format(epoch))
        if not folder.exists():
            folder.mkdir()
        img_name = os.path.split(dir_left)[1]
        img_name = img_name.split('.')[0]
        save_depth_path = os.path.join(folder, '{}_depth.png'.format(img_name))
        save_img_path = os.path.join(folder, '{}.png'.format(img_name))
        cv2.imwrite(save_depth_path, heat_cp)
        cv2.imwrite(save_img_path, left_img)
        h, w, c = depth.shape
        depth = depth.reshape([h * w, c])
        print(depth.shape)
        dir_txt = os.path.join(folder, '{}_depth.txt'.format(img_name))
        np.savetxt(dir_txt, depth, fmt='%f %f %f', delimiter='\n')

