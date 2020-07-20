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
import MyDataset as MyD
from models import submodule, gwc
from processing import display
import mytransform as transforms

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=96,
                    help='maxium disparity')
parser.add_argument('--model', default='GwcNet_GC',
                    help='select model')
parser.add_argument('--datapath', default='/home/greatbme/data_plus/Stereo Dataset',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
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
train_transform_list = [transforms.RandomColor(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
train_transform = transforms.Compose(train_transform_list)

training_transforms = albu.Compose([
        # Color augmentation
        albu.OneOf([
            albu.Compose([
                albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                # albu.RandomGamma(gamma_limit=(80, 120), p=0.01),
                albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0, p=0.5)]),
            albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5)
        ]),
        # Image quality augmentation
        albu.OneOf([
            albu.Blur(p=0.3),
            albu.MedianBlur(p=0.3, blur_limit=3),
            albu.MotionBlur(p=0.3),
        ]),
        # Noise augmentation
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 30), p=0.5),
            albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.005 * 255, 0.02 * 255), p=0.5)
        ]), ], p=0.6)

if args.model == 'GwcNet_GC':
    model = gwc.GwcNet_GC(int(args.maxdisp))
elif args.model == 'GwcNet_G':
    model = gwc.GwcNet_G(int(args.maxdisp))
else:
    print('no model')

if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

if args.loadmodel is not None:
    if Path(args.loadmodel).exists():
        print("Loading {:s} ......".format(args.loadmodel))
        state_dict = torch.load(args.loadmodel)
        step = state_dict['step']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        exit()
else:
    epoch = 0
    step = 0

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
currentDT = datetime.datetime.now()
log_root = Path(args.savemodel)/"depth_estimation_training_run_{}_{}_{}_valid_{}".format(
    currentDT.month,
    currentDT.day,
    currentDT.hour, 2)
if not log_root.exists():
    log_root.mkdir()
writer = SummaryWriter(logdir=str(log_root))
print("Tensorboard visualization at {}".format(str(log_root)))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(sample):
    model.train()
    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    depth = sample['depth']  # [B, H, W]
    mask = sample['mask']
    reproj_left = sample['reprojection']
    if args.cuda:
        left, right, depth, mask, reproj_left= left.cuda(), right.cuda(), depth.cuda(), mask.cuda(), reproj_left.cuda()
    # mask = mask > 0.6
    b, c, h, w = depth.shape
    count = b
    for i in range(b):
        if depth[i][mask[i]].numel() < depth[i].numel() * 0.1:
            mask[i] = mask[i] < 0
            count -= 1
    if count < 1:
        return -1, -1, -1, -1, -1
    optimizer.zero_grad()
    if args.model == 'GwcNet_GC':
        output = model(left, right)
        output0 = output[0]
        output1 = output[1]
        output2 = output[2]
        output3 = output[3]
        b, h, w = output1.shape
        bg, c, hg, wg = depth.shape
        output0 = output0.unsqueeze(1)
        output1 = output1.unsqueeze(1)
        output2 = output2.unsqueeze(1)
        output3 = output3.unsqueeze(1)
        ds = hg / h
        output0 = F.upsample(output0, [hg, wg], mode='bilinear')
        output1 = F.upsample(output1, [hg, wg], mode='bilinear')
        output2 = F.upsample(output2, [hg, wg], mode='bilinear')
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')

        output0 = torch.mul(output0, ds)
        output1 = torch.mul(output1, ds)  # 上采样后要乘以采样率
        output2 = torch.mul(output2, ds)
        output3 = torch.mul(output3, ds)

        output0 = torch.squeeze(output0, 1)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)  # 输出是batch*height*width

        depth0 = submodule.reprojection()(output0, reproj_left)
        depth1 = submodule.reprojection()(output1, reproj_left)
        depth2 = submodule.reprojection()(output2, reproj_left)
        depth3 = submodule.reprojection()(output3, reproj_left)

        # fn = torch.nn.MSELoss()
        loss = 0.5 * F.smooth_l1_loss(depth0[mask], depth[mask]) + 0.5 * F.smooth_l1_loss(depth1[mask], depth[mask]) + 0.7 * F.smooth_l1_loss(depth2[mask], depth[mask]) + F.smooth_l1_loss(depth3[mask], depth[mask])

    elif args.model == 'basic':
        output3 = model(ds_left, ds_right)
        output3 = output3.unsqueeze(1)
        b, d, h, w = output3.shape
        bg, c, hg, wg = rec_left_gt.shape
        ds = hg / h
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')
        output3 = torch.mul(output3, ds)
        output3 = torch.squeeze(output3, 1)
        depth3 = submodule.reprojection()(output3, reproj_left)
        loss = F.l1_loss(rec_left_gt[mask_left], depth3[mask_left], size_average=True)

    loss.backward()
    optimizer.step()
    # display.display_color_disparity_depth(step, writer, ds_left, output3.unsqueeze(1), depth3, is_return_img=False)
    return loss.item(), count, output3, depth3, mask


def Test(sample):
    model.training = False
    model.eval()

    left = sample['left']  # [B, 3, H, W]
    right = sample['right']
    depth = sample['depth']  # [B, H, W]
    mask = sample['mask']
    reproj_left = sample['reprojection']

    if args.cuda:
        left, right, mask, reproj_left = left.cuda(), right.cuda(), mask.cuda(), reproj_left.cuda()
    # mask = mask > 0.6
    b, c, h, w = depth.shape
    count = b
    for i in range(b):
        if depth[i][mask[i]].numel() < depth[i].numel() * 0.1:
            mask[i] = mask[i] < 0
            count -= 1
    if count < 1:
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    with torch.no_grad():
        output3 = model(left, right)[0]
        b, h, w = output3.shape
        bg, c, hg, wg = depth.shape
        ds = hg / h
        output3 = output3.unsqueeze(1)
        output3 = F.upsample(output3, [hg, wg], mode='bilinear')
        output3 = torch.mul(output3, ds)
        output3 = torch.squeeze(output3, 1)
        depth3 = submodule.reprojection()(output3, reproj_left)

    output3 = output3.data.cpu()
    depth3 = depth3.data.cpu()
    mask = mask.data.cpu()

    gt_z = depth[:, 2, :, :]
    cp_z = depth3[:, 2, :, :]
    z_mask = mask[:, 2, :, :]
    mae_z = torch.mean(torch.abs(gt_z[z_mask] - cp_z[z_mask]))
    mae_xyz = torch.mean(torch.abs(depth[mask] - depth3[mask]))
    bad2_mask = torch.abs(gt_z - cp_z) > 2
    bad1_mask = torch.abs(gt_z - cp_z) > 1
    bad05_mask = torch.abs(gt_z - cp_z) > 0.5
    bad10_mask = torch.abs(gt_z - cp_z) > 10
    bad2 = gt_z[bad2_mask & z_mask].numel() / gt_z[z_mask].numel()
    bad1 = gt_z[bad1_mask & z_mask].numel() / gt_z[z_mask].numel()
    bad05 = gt_z[bad05_mask & z_mask].numel() / gt_z[z_mask].numel()
    bad10 = gt_z[bad10_mask & z_mask].numel() / gt_z[z_mask].numel()
    return output3, depth3, mask, mae_xyz.item(), mae_z.item(), bad10, bad2, bad1, bad05, count


def adjust_learning_rate(optimizer, epoch):
    if epoch < 3:
        lr = 0.001
    else:
        lr = 0.0005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    batch_size = 8
    display_interval = 100
    scalar_interval = 50
    validation_interval = 1

    Traindata = MyD.MyDataset(args.datapath, transform=train_transform, downsampling=2)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=Traindata, batch_size=batch_size, shuffle=True,  drop_last=False)
    # Testdata = MyD.MyDataset(args.datapath, downsampling=2, phase='test')
    # TestImgLoader = torch.utils.data.DataLoader(dataset=Testdata, batch_size=batch_size, shuffle=False, drop_last=False)
    Validationdata = MyD.MyDataset(args.datapath, downsampling=2, phase='validation')
    ValidationImgLoader = torch.utils.data.DataLoader(dataset=Validationdata, batch_size=batch_size, shuffle=False, drop_last=False)
    start_full_time = time.time()

    for epoch in range(epoch + 1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        mean_loss = 0
        all_count = 0
        train_step = 0
        adjust_learning_rate(optimizer, epoch)
        tq = tqdm.tqdm(total=len(TrainImgLoader) * batch_size, dynamic_ncols=True, ncols=80)

        ## training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, count, disparity, depth, mask = train(sample)

            if loss < 0:
                continue
            else:
                mean_loss = (mean_loss * all_count + loss * count)/(all_count + count)
                all_count += count
            #  print('Iter %d training loss = %.3f , time = %.2f s' % (batch_idx, loss, time.time() - start_time))
            step += 1
            train_step += 1

            tq.update(batch_size)
            tq.set_postfix(loss='avg: {:.5f}  cur: {:.5f}'.format(mean_loss, loss))
            if train_step % scalar_interval == 0:
                writer.add_scalar('Training/loss',  mean_loss, step)
            if train_step % display_interval == 0:
                display.display_color_disparity_depth(epoch, train_step, writer, sample['left'], disparity, depth, sample['depth'], mask, is_return_img=False)
            if train_step > 1800:
                break
        tq.close()
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'step': step,
            'state_dict': model.state_dict(),
            'train_loss': mean_loss,
        }, savefilename)
        writer.export_scalars_to_json(str(log_root / ('all_scalars_' + str(epoch) + '.json')))

        if epoch % validation_interval != 0:
           continue

        mean_mae_xyz = 0
        mean_mae_z = 0
        mean_bad10 = 0
        mean_bad2 = 0
        mean_bad05 = 0
        mean_bad1 = 0
        all_count = 0
        val_step = 0
        tq = tqdm.tqdm(total=len(ValidationImgLoader) * batch_size, dynamic_ncols=True, ncols=80)
        tq.set_description('Validation Epoch {}'.format(epoch))
        with torch.no_grad():
            for batch_idx, sample in enumerate(ValidationImgLoader):
                output3, depth3, mask, mae_xyz, mae_z, bad10, bad2, bad1, bad05, count = Test(sample)
                if count < 0:
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
                tq.update(batch_size)
                tq.set_postfix(loss='Valid_avg: {:.5f}  Valid_cur: {:.5f}'.format(mean_mae_z, mae_z))
                # tq.set_postfix(disparity_loss='Valid_avg: {:.5f}   Valid_cur: {:.5f}'.format(valid_mean_disparity_loss, loss_d))
                val_step += 1
                # writer.add_scalar('Validation/epoch{}_mae_xyz'.format(epoch), mae_xyz, val_step)
                # writer.add_scalar('Validation/epoch{}_mae_z'.format(epoch), mae_z, val_step)
                # writer.add_scalar('Validation/epoch{}_bad10'.format(epoch), bad10, val_step)
                # writer.add_scalar('Validation/epoch{}_bad2'.format(epoch), bad2, val_step)
                # writer.add_scalar('Validation/epoch{}_bad1'.format(epoch), bad1, val_step)
                # writer.add_scalar('Validation/epoch{}_bad05'.format(epoch), bad05, val_step)
                # writer.add_scalar('Validation/epoch{}disparity_EPE'.format(epoch), loss, val_step)
                # writer.add_scalar('Validation/epoch{}_bad2_perc'.format(epoch), bad1_perc, val_step)
                if val_step % 50 == 0:
                    display.display_color_disparity_depth(epoch, val_step, writer, sample['left'], output3, depth3, sample['depth'],
                                                    mask, phase='validation', is_return_img=False)

        writer.add_scalar('Validation/mean_bad10', mean_bad10, epoch)
        writer.add_scalar('Validation/mean_bad2', mean_bad2, epoch)
        writer.add_scalar('Validation/mean_bad1', mean_bad1, epoch)
        writer.add_scalar('Validation/mean_bad05', mean_bad05, epoch)
        writer.add_scalar('Validation/mean_mae_xyz', mean_mae_xyz, epoch)
        writer.add_scalar('Validation/mean_mae_z', mean_mae_z, epoch)
        tq.close()

    print('full training time = %.2f HR' % ((time.time() - start_full_time)/3600))