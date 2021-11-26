import matplotlib.pyplot as plt
import numpy as np
import torch
from math import pi, tan, sqrt

left_pos = np.array([-0.027, 0.436, 0.385])
left_target = np.array([1.288, -0.086, -0.086])
left_up = np.array([1.19, 0.0, 0.992])

right_pos = np.array([-0.027, -0.436, 0.385])
right_target = np.array([1.288, 0.086, -0.086])
right_up = np.array([1.19, 0.0, 0.992])


K = np.zeros((3, 3), dtype='float64')
K[0, 0], K[0, 2] = 128/tan(90*pi/360), 128
K[1, 1], K[1, 2] = 128/tan(59*pi/360), 128
K[2, 2] = 1.



left_dir = left_target - left_pos
left_right = np.cross(left_dir,left_up)
left_UP = np.cross(left_dir, left_right)

left_dir = left_dir / np.linalg.norm(left_dir)
left_right = left_right / np.linalg.norm(left_right)
left_UP = left_UP / np.linalg.norm(left_UP)

left_matrix = np.zeros((4,4), dtype='float64')
left_matrix[:3,0], left_matrix[:3,1], left_matrix[:3,2], left_matrix[:3,3] \
= left_right, left_UP, left_dir, left_pos
left_matrix[3,3] = 1
left_matrix = np.linalg.inv(left_matrix)[:3,:]  # left_to_world

right_dir = right_target - right_pos
right_right = np.cross(right_dir,right_up)
right_UP = np.cross(right_dir, right_right)

right_dir = right_dir / np.linalg.norm(right_dir)
right_right = right_right / np.linalg.norm(right_right)
right_UP = right_UP / np.linalg.norm(right_UP)

right_matrix = np.zeros((4,4), dtype='float64')
right_matrix[:3,0], right_matrix[:3,1], right_matrix[:3,2], right_matrix[:3,3]\
 = right_right, right_UP, right_dir, right_pos
right_matrix[3,3] = 1
right_matrix = np.linalg.inv(right_matrix)[:3,:] # right_to_world



K = torch.from_numpy(K).cuda()
left_matrix = torch.from_numpy(left_matrix).cuda()
right_matrix = torch.from_numpy(right_matrix).cuda()


def make_heat_map(heat, u,v, num):
	
	for i in range(v-2,v+3):
		for j in range(u-2, u+3):
			dist = np.linalg.norm([v-i, u-j], axis=-1)

			if i < 0 or i > heat.shape[0] or j < 0 or j > heat.shape[1]:
				continue

			heat[0,num,i,j]= int(255*(1-dist/sqrt(8)))

	return heat

def get_uv(key_list):

	heat_map_l = np.zeros((256,256,len(key_list)), dtype='float64')
	heat_map_r = np.zeros((256,256,len(key_list)), dtype='float64')

	uv_l = []
	uv_r = []

	for i in range(len(key_list)):	

		target_pos = key_list[i]

		target_pos = np.concatenate((target_pos, (1,))) # x,y,z,1

		left_coor = np.matmul(left_matrix, target_pos.T) # left to world * world vector
		left_uv = np.matmul(K,left_coor)
		left_uv = (1/left_uv[2])*left_uv

		right_coor = np.matmul(right_matrix, target_pos.T)
		right_uv = np.matmul(K,right_coor)
		right_uv = (1/right_uv[2])*right_uv

		u1 = int(left_uv[0])
		v1 = int(left_uv[1])

		u2 = int(right_uv[0])
		v2 = int(right_uv[1])

		heat_map_l = make_heat_map(heat_map_l, u1, v1, i)
		heat_map_r = make_heat_map(heat_map_r, u2, v2, i)

		uv_l.append([u1,v1])
		uv_r.append([u2,v2])


	return heat_map_l, heat_map_r, uv_l, uv_r

def cal_xyz(l_uv, r_uv):

	M1 = torch.matmul(K, left_matrix) # K *  left to world
	M2 = torch.matmul(K, right_matrix) # K * right to world
	pre = None

	for i in range(l_uv.shape[0]):

		A = torch.cat(((l_uv[i][1]*M1[2,:] - M1[0,:]).unsqueeze(0), (l_uv[i][0]*M1[2,:] - M1[1,:]).unsqueeze(0), (r_uv[i][1]*M2[2,:] - M2[0,:]).unsqueeze(0),(r_uv[i][0]*M2[2,:] - M2[1,:]).unsqueeze(0)
  ), dim=0).cuda()
		_, _, s3 = torch.svd(A)

		s3 = s3/s3[3]

		if pre is None:
			pre = s3[:3,3].unsqueeze(0).unsqueeze(0)
		else : 
			pre = torch.cat((pre, s3[:3,3].unsqueeze(0).unsqueeze(0)), dim=1)

	return pre

def projection(key_list, img_info):

	heat_map_l = torch.zeros((1,len(key_list),img_info[4],img_info[5]))
	heat_map_r = torch.zeros((1,len(key_list),img_info[4],img_info[5]))

	l_offset_v = img_info[0] - int(img_info[4]/2.0)
	l_offset_u = img_info[1] - int(img_info[5]/2.0)
	r_offset_v = img_info[2] - int(img_info[4]/2.0)
	r_offset_u = img_info[3] - int(img_info[5]/2.0)
	
	for i in range(len(key_list)):	

		target_pos = torch.ones((4,), dtype=torch.float64)

		target_pos[:3] = key_list[0,i,:]

		left_coor = torch.matmul(left_matrix, target_pos.T) # left to world * world vector
		left_uv = torch.matmul(K,left_coor)
		left_uv = (1/left_uv[2])*left_uv

		right_coor = torch.matmul(right_matrix, target_pos.T)
		right_uv = torch.matmul(K,right_coor)
		right_uv = (1/right_uv[2])*right_uv

		u1 = int(left_uv[0]) - l_offset_u
		v1 = int(left_uv[1]) - l_offset_v

		u2 = int(right_uv[0]) - r_offset_u
		v2 = int(right_uv[1]) - r_offset_v

		heat_map_l = make_heat_map(heat_map_l, u1, v1, i)
		heat_map_r = make_heat_map(heat_map_r, u2, v2, i)

	return heat_map_l, heat_map_r


def cal_real_uv(pre_uv, img_info):

	# real_uv = pre_uv
	off_set_v = (img_info[0] - img_info[2]/2.0).cuda().type(torch.float64)
	off_set_u = (img_info[1] - img_info[3]/2.0).cuda().type(torch.float64)
	pre_uv[:,0] += off_set_v
	pre_uv[:,1] += off_set_u

	return pre_uv


def cal_uv(heat, key_num):

	C,H,W = heat.shape

	key_heat = np.argmax(np.reshape(heat, (C,-1)),axis=1)
	y_list = key_heat%W
	x_list = np.floor(key_heat/W)%H

	uv_list = np.zeros((key_num,2))
	for i in range(key_num):
		uv_list[i,0] = x_list[i]
		uv_list[i,1] = y_list[i]

	return uv_list

def draw_heat(image, uv_list, key_num):


	heat_map = np.zeros((image.shape))
	for i in range(key_num):
		for v in range(int(uv_list[i,0])-3,int(uv_list[i,0])+4):
			for u in range(int(uv_list[i,1])-3,int(uv_list[i,1])+4):
				dist = np.linalg.norm([uv_list[i,0] -v, uv_list[i,1]-u], axis=-1)
				if i <3:	
					heat_map[v,u,i] = int(255*(1-dist/sqrt(18)))
				else:
					heat_map[v,u,i-3] = int(255*(1-dist/sqrt(18)))
					heat_map[v,u,i-2] = int(255*(1-dist/sqrt(18)))
				# heat_map[v,u,0] = 255
	image = 255*image + heat_map
	image = np.clip(image, 0,255)
	return image

def np_cal_xyz(l_uv, r_uv, l_K, r_K, l2w, r2w):

	M1 = np.matmul(l_K, l2w) # K *  left to world
	M2 = np.matmul(r_K, r2w) # K * right to world
	
	A = np.zeros((4,4))

	target_pos = []

	for i in range(l_uv.shape[0]):

		A[0,:] = l_uv[i][1]*M1[2,:] - M1[0,:]
		A[1,:] = l_uv[i][0]*M1[2,:] - M1[1,:]
		A[2,:] = r_uv[i][1]*M2[2,:] - M2[0,:]
		A[3,:] = r_uv[i][0]*M2[2,:] - M2[1,:]

		_, _, s3 = np.svd(A)

		s3 = s3.T/s3.T[3]

		target_pos.append(s3[:3,3])

	return np.array(target_pos)