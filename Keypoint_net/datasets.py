# import glob
import random
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages/')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from natsort import natsorted
from shadow_aug import *


class ImageDataset(Dataset):
	def __init__(self, mode="train"):
		if mode == "train":
			self.data_path = '../data/many'
		elif mode == "val":
			self.data_path = '../data/many_val'

		self.left_img_path = os.path.join(self.data_path, 'left/crop')
		self.left_heat_path = os.path.join(self.data_path, 'left/heat')
		self.left_info_path = os.path.join(self.data_path, 'left/zero_size')
		self.left_uv_path = os.path.join(self.data_path, 'left/uv')
		self.left_depth_path = os.path.join(self.data_path, 'left/depth')
		self.left_img_list = natsorted(os.listdir(self.left_img_path))

		self.right_img_path = os.path.join(self.data_path, 'right/crop')
		self.right_heat_path = os.path.join(self.data_path, 'right/heat')
		self.right_info_path = os.path.join(self.data_path, 'right/zero_size')
		self.right_uv_path = os.path.join(self.data_path, 'right/uv')
		self.right_depth_path = os.path.join(self.data_path, 'right/depth')
		self.right_img_list = natsorted(os.listdir(self.right_img_path))

		self.key_list_path = os.path.join(self.data_path, 'keypoint')

	def __getitem__(self, index):

		# Data load
		left_target = self.left_img_list[index]
		right_target = self.right_img_list[index]
		key_target = left_target.replace('left','keypoint')

		# Crop img
		img_left = np.load('%s/%s'%(self.left_img_path, left_target))
		img_right = np.load('%s/%s'%(self.right_img_path, right_target))

		# Heatmap
		heat_left = np.load('%s/%s'%(self.left_heat_path, left_target))
		heat_right = np.load('%s/%s'%(self.right_heat_path, right_target))

		# Depth
		depth_left = np.load('%s/%s'%(self.left_depth_path, left_target))
		depth_right = np.load('%s/%s'%(self.right_depth_path, right_target))

		# Image info
		info_left = np.load('%s/%s'%(self.left_info_path, left_target))
		info_right = np.load('%s/%s'%(self.right_info_path, right_target))

		# UV coord
		uv_left = np.load('%s/%s'%(self.left_uv_path, left_target))
		uv_right = np.load('%s/%s'%(self.right_uv_path, right_target))

		# 3D Keypoint
		keypoints = np.load('%s/%s'%(self.key_list_path, key_target))        


		# Data preprocess
		img_left = add_noise(img_left[:,:,:3])
		eps1 = np.random.uniform()
		if eps1 < 0.5:
			img_left = add_circle(img_left)
		else:
			img_left = add_shadow(img_left)

		img_right = add_noise(img_right[:,:,:3])
		eps1 = np.random.uniform()
		if eps1 < 0.5:
			img_right = add_circle(img_right)
		else:
			img_right = add_shadow(img_right)


		input_image = np.concatenate([img_left,img_right],axis=2)
		con_depth = np.concatenate([depth_left,depth_right],axis=2)

		input_image, con_depth, heat_left, heat_right = np.transpose(input_image,[2, 0, 1]), np.transpose(con_depth,[2, 0, 1]), np.transpose(heat_left,[2, 0, 1]), np.transpose(heat_right,[2, 0, 1])

		input_image = torch.FloatTensor(input_image)/255.0
		con_depth = torch.FloatTensor(con_depth)
		heat_left = torch.FloatTensor(heat_left)/255.0
		heat_right = torch.FloatTensor(heat_right)/255.0

		return {"input" : input_image, "depth" : con_depth, "h_left" : heat_left, "h_right" : heat_right,
		 "l_size" :  info_left, "r_size" :  info_right, "key_list" : keypoints, "uv_l" : uv_left, "uv_r" : uv_right}
		

	def __len__(self):
		return len( self.left_img_list)
