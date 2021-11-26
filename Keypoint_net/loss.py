import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

criterion_BCE = torch.nn.BCELoss()
criterion_KLD = torch.nn.KLDivLoss()
criterion_mse = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_cross = torch.nn.CrossEntropyLoss(reduction="sum")
cosine_similarity = nn.CosineSimilarity(dim=-1)

cuda = True if torch.cuda.is_available() else False

if cuda:
	criterion_BCE.cuda()
	criterion_KLD.cuda()
	criterion_L1.cuda()
	criterion_mse.cuda()
	criterion_cross.cuda()
	cosine_similarity.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

KEY_NUM = int(input("Key num ?"))

def gen_loss(l1,l2,l3,r1,r2,r3):

	N,C,H,W = l1.shape

	valid = Variable(Tensor(np.ones((N,1,H,W))), requires_grad=False)
	fake = Variable(Tensor(np.zeros((N,1,H,W))), requires_grad=False)

	l1_loss = criterion_mse(l1, valid)        
	l2_loss = criterion_mse(l2, valid)        
	l3_loss = criterion_mse(l3, valid)

	r1_loss = criterion_mse(r1, valid)        
	r2_loss = criterion_mse(r2, valid)        
	r3_loss = criterion_mse(r3, valid)

	return l1_loss+l2_loss+l3_loss+r1_loss+r2_loss+r3_loss

def pixel_loss(heat, fin_heat, depth, rgb, gt_heat, gt_depth, gt_rgb, key_num=5):

	dep_loss = criterion_L1(depth, gt_depth)
	rgb_loss = criterion_L1(rgb, gt_rgb)
	heat_loss = criterion_BCE(heat[:,:KEY_NUM,:,:], gt_heat[:,:KEY_NUM,:,:])
	fin_heat_loss = criterion_BCE(fin_heat[:,:KEY_NUM,:,:], gt_heat[:,:KEY_NUM,:,:])

	return dep_loss+rgb_loss+heat_loss+2*fin_heat_loss


def dis_loss(r1,r2,r3,f1,f2,f3):

	N,C,H,W = r1.shape

	valid = Variable(Tensor(np.ones((N,1,H,W))), requires_grad=False)
	fake = Variable(Tensor(np.zeros((N,1,H,W))), requires_grad=False)

	r1_loss = criterion_mse(r1, valid)        
	r2_loss = criterion_mse(r2, valid)        
	r3_loss = criterion_mse(r3, valid)

	f1_loss = criterion_mse(f1, fake)        
	f2_loss = criterion_mse(f2, fake)        
	f3_loss = criterion_mse(f3, fake)

	return r1_loss+r2_loss+r3_loss+f1_loss+f2_loss+f3_loss


def contrastive_loss(key_vector1, clutter_vector1, key_vector2, clutter_vector2):

	representation = torch.cat((key_vector1,clutter_vector1,key_vector2,clutter_vector2),axis=0)
	similar_matrix = cosine_similarity(representation.unsqueeze(1),representation.unsqueeze(0))

	a = torch.diag(similar_matrix, 10)[:5]
	b = torch.diag(similar_matrix, -10)[:5]

	positive = torch.cat([a,b]).view(10,1)

	diag = np.eye(2 *10)
	l1 = np.eye((2 *10), 2 *10, k=-10)
	l2 = np.eye((2 *10), 2 *10, k= 10)
	l3 = np.zeros((20,20))
	mi = np.ones((10,10))-np.eye(10)
	l3[10:,10:] = mi

	mask = torch.from_numpy((diag+l1+l2+l3))
	mask = (1-mask).type(torch.bool).cuda()

	negative = similar_matrix[mask].view(10,-1)

	logits = torch.cat((positive,negative),dim=1)
	logits /= 0.5

	labels = torch.zeros(10).cuda().long()

	loss = criterion_cross(logits,labels)

	return loss

def cal_uv_loss_train(left, right, gt_left, gt_right,H,W):

	gt_left[:,0] = gt_left[:,0]/float(W) - 0.5
	gt_left[:,1] = gt_left[:,1]/float(H) - 0.5
	gt_right[:,0] = gt_right[:,0]/float(W) - 0.5
	gt_right[:,1] = gt_right[:,1]/float(H) - 0.5

	l_loss = criterion_mse(left,gt_left.cuda())
	r_loss = criterion_mse(right,gt_right.cuda())

	return l_loss + r_loss

def cal_uv_loss(left, right, gt_left, gt_right):

	l_loss = criterion_mse(left,gt_left.cuda())
	r_loss = criterion_mse(right,gt_right.cuda())

	return l_loss + r_loss

def cal_key_loss(key_list, gt_key_list):

	return criterion_mse(key_list,gt_key_list.type(torch.float64).cuda())


