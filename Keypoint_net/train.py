import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image

from models import *
from datasets import *
from loss import *

from camera_cal import * 
from soft_argmax import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="keypoint_net", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between model checkpoints")
parser.add_argument("--object", type=str, default='drawer', help="target object")
opt = parser.parse_args()
print(opt)

if opt.object == 'drawer':
    KEY_NUM, CLUTTER_NUM = 5, 5
elif  opt.object == 'mug':
    KEY_NUM, CLUTTER_NUM = 3, 3

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
keypoint_net = KeypointNet(key_num=KEY_NUM, clutter_num= CLUTTER_NUM)
discriminator_heat = Discriminator(dex='heat', key_num=KEY_NUM)
discriminator_depth = Discriminator(dex='dep', key_num=KEY_NUM)
discriminator_rgb = Discriminator(dex='rgb', key_num=KEY_NUM)

if cuda:
    keypoint_net = keypoint_net.cuda()
    discriminator_heat = discriminator_heat.cuda()
    discriminator_depth = discriminator_depth.cuda()
    discriminator_rgb = discriminator_rgb.cuda() 

if opt.epoch != 0:
    # Load pretrained models
    keypoint_net.load_state_dict(torch.load("saved_models/%s/keypoint_net%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_heat.load_state_dict(torch.load("saved_models/%s/discriminator_heat%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_depth.load_state_dict(torch.load("saved_models/%s/discriminator_depth%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_rgb.load_state_dict(torch.load("saved_models/%s/discriminator_rgb%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    keypoint_net.apply(weights_init_normal)
    discriminator_heat.apply(weights_init_normal)
    discriminator_depth.apply(weights_init_normal)
    discriminator_rgb.apply(weights_init_normal)

# Optimizers
optimizer = torch.optim.Adam(keypoint_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_heat = torch.optim.Adam(discriminator_heat.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_depth = torch.optim.Adam(discriminator_depth.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_rgb = torch.optim.Adam(discriminator_rgb.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders

dataloader = DataLoader(
    ImageDataset(),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset(mode="val"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done, key_num):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    input_img = imgs["input"].cuda()
    h_left = imgs["h_left"].cuda()
    h_right = imgs["h_right"].cuda()
    dep_img = imgs["depth"].cuda()

    left_rgb, right_rgb = input_img[:,:3,:,:], input_img[:,3:,:,:]
    visu_dep_l, visu_dep_r = dep_img[:,:1,:,:], dep_img[:,1:,:,:]

    r_dep = torch.cat((visu_dep_l,visu_dep_r),axis=3)
    r_dep = torch.cat((r_dep,r_dep,r_dep),axis=1)

    middle_heat_l, rgb_l, depth_l, final_heat_l, middle_heat_r, rgb_r, depth_r, final_heat_r = keypoint_net(left_rgb, right_rgb, mode="val")

    f_dep = torch.cat((depth_l,depth_r),axis=3)
    f_dep = torch.cat((f_dep,f_dep,f_dep),axis=1)

    in_visu = torch.cat((left_rgb, right_rgb),axis=3)
    visu_heat1 = torch.cat((h_left[:,:3,:,:],h_right[:,:3,:,:]),axis=3)
    gen_visu_heat1 = torch.cat((final_heat_l[:,:3,:,:],final_heat_r[:,:3,:,:]),axis=3)
    img_sample = torch.cat((in_visu.data,r_dep.data, f_dep.data,visu_heat1.data,, gen_visu_heat1.data), -2)
    if key_num > 3:
        visu_heat2 = torch.cat((h_left[:,2:5,:,:],h_right[:,2:5,:,:]),axis=3)
        gen_visu_heat2 = torch.cat((final_heat_l[:,2:5,:,:],final_heat_r[:,2:5,:,:]),axis=3)
        img_sample = torch.cat((in_visu.data,r_dep.data, f_dep.data,visu_heat1.data,visu_heat2.data, gen_visu_heat1.data,gen_visu_heat2.data), -2)

    
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()

iterr = 0
batch_size = 10

loss_G = 0
loss_D = 0
contra_loss = 0
uv_schedule = 0 

batch_repre_l = torch.FloatTensor().cuda()
batch_repre_r = torch.FloatTensor().cuda()


for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        input_img = batch["input"].cuda()
        dep_img = batch["depth"].cuda()

        h_left = batch["h_left"].cuda()
        h_right = batch["h_right"].cuda()

        l_size = batch["l_size"].squeeze(0)
        r_size = batch["r_size"].squeeze(0)

        true_key_list = batch["key_list"]

        uv_l = batch["uv_l"].squeeze(0)
        uv_r = batch["uv_r"].squeeze(0)

        left_rgb, right_rgb = input_img[:,:3,:,:], input_img[:,3:,:,:]
        left_depth, right_depth = dep_img[:,:1,:,:], dep_img[:,1:,:,:]

        B,C,H,W = left_rgb.shape

        off_set_v = (l_size[0] - l_size[2]/2.0).cuda().type(torch.float64)
        off_set_u = (l_size[1] - l_size[3]/2.0).cuda().type(torch.float64)
        uv_l[:,0] -= off_set_v
        uv_l[:,1] -= off_set_u

        off_set_v = (r_size[0] - r_size[2]/2.0).cuda().type(torch.float64)
        off_set_u = (r_size[1] - r_size[3]/2.0).cuda().type(torch.float64)
        uv_r[:,0] -= off_set_v
        uv_r[:,1] -= off_set_u

        uv_list = torch.cat((uv_l,uv_r),axis=0)


        # ------------------
        #  Train Generators
        # ------------------       

        # Contrastive loss
        # if i%2 == 0:
        #     key_vector1, clutter_vector1, middle_heat_l, rgb_l, depth_l, final_heat_l, middle_heat_r, rgb_r, depth_r, final_heat_r = keypoint_net(left_rgb, right_rgb, uv_list=uv_list)

        # elif i%2 == 1:
        #     key_vector2, clutter_vector2, middle_heat_l, rgb_l, depth_l, final_heat_l, middle_heat_r, rgb_r, depth_r, final_heat_r = keypoint_net(left_rgb, right_rgb, uv_list=uv_list)
    
        pred_fake_heat_l = discriminator_heat(middle_heat_l[:,:KEY_NUM,:,:],left_rgb)
        pred_fake_depth_l = discriminator_depth(depth_l,left_rgb)
        pred_fake_rgb_l = discriminator_rgb(rgb_l,left_rgb)

        pred_fake_heat_r = discriminator_heat(middle_heat_r[:,:KEY_NUM,:,:],left_rgb)
        pred_fake_depth_r = discriminator_depth(depth_r,left_rgb)
        pred_fake_rgb_r = discriminator_rgb(rgb_r,left_rgb)

        generator_loss = gen_loss(pred_fake_heat_l, pred_fake_depth_l, pred_fake_rgb_l, pred_fake_heat_r, pred_fake_depth_r, pred_fake_rgb_r)
        left_pix_loss = pixel_loss(middle_heat_l, final_heat_l, depth_l, rgb_l, h_left, left_depth, left_rgb, key_num=KEY_NUM)
        right_pix_loss = pixel_loss(middle_heat_r, final_heat_r, depth_r, rgb_r, h_right, right_depth, right_rgb, key_num=KEY_NUM)        
        
        loss_G += 0.5*generator_loss+ 50*(left_pix_loss+right_pix_loss)

        #if i%2 == 1:
        #    contra_loss = contrastive_loss(key_vector1, clutter_vector1, key_vector2, clutter_vector2)
        #    loss_G += 0.1*contra_loss


        # ------------------
        #  Train Discriminators
        # ------------------ 
        
        pred_real_heat_l = discriminator_heat(h_left[:,:KEY_NUM,:,:], left_rgb)
        pred_real_depth_l = discriminator_depth(left_depth, left_rgb)
        pred_real_rgb_l = discriminator_rgb(left_rgb, left_rgb)

        pred_fake_heat_l = discriminator_heat(middle_heat_l[:,:KEY_NUM,:,:].detach(),left_rgb)
        pred_fake_depth_l = discriminator_depth(depth_l.detach(),left_rgb)
        pred_fake_rgb_l = discriminator_rgb(rgb_l.detach(),left_rgb)

        pred_real_heat_r = discriminator_heat(h_right[:,:KEY_NUM,:,:], right_rgb)
        pred_real_depth_r = discriminator_depth(right_depth, right_rgb)
        pred_real_rgb_r = discriminator_rgb(right_rgb, right_rgb)

        pred_fake_heat_r = discriminator_heat(middle_heat_r[:,:KEY_NUM,:,:].detach(),right_rgb)
        pred_fake_depth_r = discriminator_depth(depth_r.detach(),right_rgb)
        pred_fake_rgb_r = discriminator_rgb(rgb_r.detach(),right_rgb)

        left_dis_loss = dis_loss(pred_real_heat_l,pred_real_depth_l, pred_real_rgb_l, pred_fake_heat_l, pred_fake_depth_l, pred_fake_rgb_l)
        right_dis_loss = dis_loss(pred_real_heat_r,pred_real_depth_r, pred_real_rgb_r, pred_fake_heat_r, pred_fake_depth_r, pred_fake_rgb_r)

        loss_D += 0.5*0.5*(left_dis_loss + right_dis_loss)


        # Cal xyz 
        pred_left_uv = softmax_integral_tensor(final_heat_l[:,:KEY_NUM,:,:]).squeeze(0)
        pred_right_uv = softmax_integral_tensor(final_heat_r[:,:KEY_NUM,:,:]).squeeze(0)

        left_indices = soft_argmax(final_heat_l[:,:KEY_NUM,:,:]).squeeze(0)
        right_indices = soft_argmax(final_heat_r[:,:KEY_NUM,:,:]).squeeze(0)

        left_uv = cal_real_uv(left_indices, l_size)
        right_uv = cal_real_uv(right_indices, r_size)
        key_list = cal_xyz(left_uv, right_uv) 

        # Cal key loss
        uv_loss = cal_uv_loss_train(pred_left_uv, pred_right_uv, uv_l, uv_r, H,W)
        uv_difference = cal_uv_loss(left_indices,right_indices,uv_l,uv_r)               
        key_loss = cal_key_loss(key_list,true_key_list)

        iterr +=1

        if iterr == batch_size:

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

            optimizer_D_heat.zero_grad()
            optimizer_D_depth.zero_grad()
            optimizer_D_rgb.zero_grad()
            loss_D.backward()
            optimizer_D_heat.step()
            optimizer_D_depth.step()
            optimizer_D_rgb.step()            

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [BCE loss: %f ] [Key loss: %f ] [UV loss: %f ] [Gen loss : %f ]  ETA: %s" # [Dep loss: %f ]
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    (left_pix_loss+right_pix_loss).item(),
                    key_loss.item(),
                    uv_difference.item(),
                    #contra_loss.item(),
                    (generator_loss+left_dis_loss + right_dis_loss).item(),
                    time_left,
                )
            )

            iterr = 0
            loss_G = 0
            loss_D = 0
            contra_loss = 0

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, key_num=KEY_NUM)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(keypoint_net.state_dict(), "saved_models/%s/keypoint_net_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_heat.state_dict(), "saved_models/%s/discriminator_heat_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_depth.state_dict(), "saved_models/%s/discriminator_depth_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_rgb.state_dict(), "saved_models/%s/discriminator_rgb_%d.pth" % (opt.dataset_name, epoch))
