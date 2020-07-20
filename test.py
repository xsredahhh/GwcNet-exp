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
import gwc
from models import basic, stackhourglass, submodule
from processing import display
import json

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=96,
                    help='maxium disparity')
parser.add_argument('--model', default='GwcNet_GC',
                    help='select model')
parser.add_argument('--datapath', default='/home/greatbme/data_plus/Stereo Dataset',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='/home/greatbme/mydata/data_plus/Experiments/PSM_CC/depth_estimation_training_run_7_7_18_valid_2/checkpoint_8.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    step = state_dict['step']
    epoch = state_dict['epoch']
    model.load_state_dict(state_dict['state_dict'])
    print('Restored model, epoch {}, step {}'.format(epoch, step))

# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

log_root = Path('/home/greatbme/mydata/data_plus/Experiments/PSMNet_parameter_comp/without_data_augmentation/depth_estimation_training_run_7_7_0_valid_2/validation{}.json'.format(epoch))

# log_root = Path('./run_test_{}'.format(epoch))
# if not log_root.exists():
#     log_root.mkdir()
# writer = SummaryWriter(logdir=str(log_root))
# print("Tensorboard visualization at {}".format(str(log_root)))


def Test(ds_left, ds_right, rec_left_gt, mask_left,  reproj_left):
    model.eval()

    if args.cuda:
        ds_left, ds_right, rec_left_gt, mask_left, reproj_left = ds_left.cuda(), ds_right.cuda(), rec_left_gt.cuda(), mask_left.cuda(), reproj_left.cuda()

    mask_left = mask_left > 0.6
    b, c, h, w = rec_left_gt.shape
    count = b
    for i in range(b):
        if rec_left_gt[i][mask_left[i]].numel() < rec_left_gt[i].numel() * 0.1:
            mask_left[i] = mask_left[i] < 0
            count -= 1
    if count < 1:
        return -1, -1, -1, -1, -1, -1, -1
    # ds_left = ds_left.unsqueeze(0)
    # ds_right = ds_right.unsqueeze(0)
    # reproj_left = reproj_left.unsqueeze(0)

    with torch.no_grad():
        output3 = model(ds_left, ds_right)[0]
        b, h, w = output3.shape
        b, c, hg, wg = rec_left_gt.shape
        ds = hg / h

        output3 = output3.unsqueeze(1)
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')
        output3 = torch.mul(output3, ds)
        output3 = torch.squeeze(output3, 1)
        depth3 = submodule.reprojection()(output3, reproj_left)

    gt_z = rec_left_gt[:, 2, :, :]
    cp_z = depth3[:, 2, :, :]
    z_mask = mask_left[:, 2, :, :]
    mae_z = torch.mean(torch.abs(gt_z[z_mask] - cp_z[z_mask]))
    mae_xyz = torch.mean(torch.abs(rec_left_gt[mask_left] - depth3[mask_left]))
    bad2_mask = torch.abs(gt_z - cp_z) > 2
    bad1_mask = torch.abs(gt_z - cp_z) > 1
    bad05_mask = torch.abs(gt_z - cp_z) > 0.5
    bad10_mask = torch.abs(gt_z - cp_z) > 10
    bad2 = gt_z[bad2_mask & z_mask].numel()/gt_z[z_mask].numel()
    bad1 = gt_z[bad1_mask & z_mask].numel()/gt_z[z_mask].numel()
    bad05 = gt_z[bad05_mask & z_mask].numel()/gt_z[z_mask].numel()
    bad10 = gt_z[bad10_mask & z_mask].numel()/gt_z[z_mask].numel()
    return mae_xyz.item(), mae_z.item(), bad10, bad2, bad1, bad05, count
    # depth3 = depth3.squeeze(0).data.cpu()
    # mae = torch.mean(torch.abs(depth3[mask_left] - rec_left_gt[mask_left]))
    # gt_z = rec_left_gt[2].unsqueeze(0).numpy()
    # cp_z = depth3[2].unsqueeze(0).numpy()
    # mask_left = mask_left.numpy()[2][np.newaxis, :]
    #
    # mid_gt = np.median(gt_z[mask_left])
    # min_gt = np.min(gt_z[mask_left])
    # max_gt = np.max(gt_z[mask_left])
    # min_diff = np.min([max_gt - mid_gt, mid_gt - min_gt])
    # min_val = max((mid_gt - 10 * min_diff), min_gt)
    # max_val = min((mid_gt + 10 * min_diff), max_gt)
    # min_val = min_gt + 10
    # max_val = max_gt - 10
    # mask_gt = (gt_z < min_val) | (gt_z > max_val)
    # mask_cp = (cp_z < min_val) | (cp_z > max_val)
    # gt_z = (gt_z - min_val) / (max_val - min_val)
    # cp_z = (cp_z - min_val) / (max_val - min_val)
    # gt_z[mask_gt] = 0
    # cp_z[mask_cp] = 0
    # heat_gt = cv2.applyColorMap(np.uint8(255 * np.moveaxis(gt_z, source=[0, 1, 2], destination=[2, 0, 1])), cv2.COLORMAP_JET)
    # heat_cp = cv2.applyColorMap(np.uint8(255 * np.moveaxis(cp_z, source=[0, 1, 2], destination=[2, 0, 1])), cv2.COLORMAP_JET)
    # print("heat_gt.shape: ", heat_gt.shape, "mask.shape: ", mask_cp.shape)
    # mask_gt = np.moveaxis(mask_gt, source=[0, 1, 2], destination=[2, 0, 1]).repeat(3, axis=2)
    # mask_cp = np.moveaxis(mask_cp, source=[0, 1, 2], destination=[2, 0, 1]).repeat(3, axis=2)
    # heat_gt[mask_gt] = 0
    # heat_cp[mask_cp] = 0
    # cv2.imwrite('./test/test{}_gt.png'.format(id), heat_gt)
    # cv2.imwrite('./test/test{}_cp.png'.format(id), heat_cp)
    # rgb = ds_left.data.cpu().squeeze(0).numpy()
    # rgb = rgb * 0.5 + 0.5
    # rgb = np.uint8(255 * np.moveaxis(rgb, source=[0, 1, 2], destination=[2, 0, 1]))
    # cv2.imwrite('./test/test{}_rgb.png'.format(id), rgb)
    # # print("mid: ", mid_gt, " min: ", min_gt, " max:", max_gt)
    # # print(gt_z.shape, cp_z.shape)
    # print("mae: ", mae)
    # return 0


if __name__ == '__main__':
    Validationdata = MyD.MyDataset(args.datapath, downsampling=2, phase='validation')
    ValidationImgLoader = torch.utils.data.DataLoader(dataset=Validationdata, batch_size=8, shuffle=False,
                                                      drop_last=False)
    mean_mae_xyz = 0
    mean_mae_z = 0
    mean_bad10 = 0
    mean_bad2 = 0
    mean_bad05 = 0
    mean_bad1 = 0
    all_count = 0
    val_step = 0
    for batch_idx, (ds_left, ds_right, rec_left_gt, mask_left, reproj_left) in enumerate(ValidationImgLoader):
        mae_xyz, mae_z, bad10, bad2, bad1, bad05, count = Test(ds_left, ds_right, rec_left_gt, mask_left,  reproj_left)

        if mae_xyz < 0:
            continue
        else:
            mean_mae_xyz = (mean_mae_xyz * all_count + mae_xyz * count) / (all_count + count)
            mean_mae_z = (mean_mae_z * all_count + mae_z * count) / (all_count + count)
            mean_bad10 = (mean_bad10 * all_count + bad10 * count) / (all_count + count)
            mean_bad2 = (mean_bad2 * all_count + bad2 * count) / (all_count + count)
            mean_bad1 = (mean_bad1 * all_count + bad1 * count) / (all_count + count)
            mean_bad05 = (mean_bad05 * all_count + bad05 * count) / (all_count + count)
            all_count += count
            #  print('Iter %d training loss = %.3f , time = %.2f s' % (batch_idx, loss, time.time() - start_time))

            # tq.set_postfix(disparity_loss='Valid_avg: {:.5f}   Valid_cur: {:.5f}'.format(valid_mean_disparity_loss, loss_d))
            val_step += 1
            # writer.add_scalar('Validation/epoch{}_mae_xyz'.format(epoch), mae_xyz, val_step)
            # writer.add_scalar('Validation/epoch{}_mae_z'.format(epoch), mae_z, val_step)
            # writer.add_scalar('Validation/epoch{}_bad10'.format(epoch), bad10, val_step)
            # writer.add_scalar('Validation/epoch{}_bad2'.format(epoch), bad2, val_step)
            # writer.add_scalar('Validation/epoch{}_bad1'.format(epoch), bad1, val_step)
            # writer.add_scalar('Validation/epoch{}_bad05'.format(epoch), bad05, val_step)

    print('mean_bad10: ', mean_bad10, 'mean_bad2: ', mean_bad2, 'mean_bad1: ', mean_bad1, 'mean_bad05: ', mean_bad05, 'mean_mae_xyz: ', mean_mae_xyz, 'mean_mae_z: ', mean_mae_z)
    dictObj = {'epoch': epoch, 'mean_bad10': mean_bad10, 'mean_bad2': mean_bad2, 'mean_bad1': mean_bad1, 'mean_bad05': mean_bad05,'mean_mae_xyz': mean_mae_xyz, 'mean_mae_z':mean_mae_z}

    jsObj = json.dumps(dictObj)
    fileObject = open(log_root, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    # writer.add_scalar('epoch{}_mean_bad10'.format(epoch), mean_bad10, 1)
    # writer.add_scalar('epoch{}_mean_bad2'.format(epoch), mean_bad2, 1)
    # writer.add_scalar('epoch{}_mean_bad1'.format(epoch), mean_bad1, 1)
    # writer.add_scalar('epoch{}_mean_bad05'.format(epoch), mean_bad05, 1)
    # writer.add_scalar('epoch{}_mean_mean_mae_xyz'.format(epoch), mean_mae_xyz, 1)
    # writer.add_scalar('epoch{}_mean_mean_mae_z'.format(epoch), mean_mae_z, 1)

    # id = 2880
    # ds_left, ds_right, rec_left_gt, rec_right_gt, mask_left, mask_right,  reproj_left, reproj_right, bb, ff = Validationdata.__getitem__(id)
    # print(ds_left.dtype, rec_left_gt.shape, reproj_left, bb, ff)
    # recall = Test(ds_left, ds_right, rec_left_gt, rec_right_gt, mask_left, mask_right,  reproj_left, reproj_right, bb, ff, id)
    #



