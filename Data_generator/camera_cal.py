import matplotlib.pyplot as plt
import numpy as np
from math import pi, tan, sqrt
import random

class camera_calculation():

	def __init__(self, key_num=3):
		super(camera_calculation, self).__init__()

		self.KEY_NUM = key_num
		self.init_camera()


	def init_camera(self):

		self.K = np.zeros((3, 3), dtype='float64')
		self.K[0, 0], self.K[0, 2] = 128/tan(90*pi/360), 128
		self.K[1, 1], self.K[1, 2] = 128/tan(59*pi/360), 128
		self.K[2, 2] = 1.

		left_pos = np.array([-0.027, 0.436, 0.385])
		left_target = np.array([1.288, -0.086, -0.086])
		left_up = np.array([1.19, 0.0, 0.992])

		right_pos = np.array([-0.027, -0.436, 0.385])
		right_target = np.array([1.288, 0.086, -0.086])
		right_up = np.array([1.19, 0.0, 0.992])

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

		self.left_matrix = np.linalg.inv(left_matrix)[:3,:]  # left_to_world

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

		self.right_matrix = np.linalg.inv(right_matrix)[:3,:] # right_to_world

	def cal_matrix(self, cam_set):

		left_dir = cam_set[1] - cam_set[0]
		left_right = np.cross(left_dir,cam_set[2])
		left_UP = np.cross(left_dir, left_right)

		left_dir = left_dir / np.linalg.norm(left_dir)
		left_right = left_right / np.linalg.norm(left_right)
		left_UP = left_UP / np.linalg.norm(left_UP)

		left_matrix = np.zeros((4,4), dtype='float64')
		left_matrix[:3,0], left_matrix[:3,1], left_matrix[:3,2], left_matrix[:3,3] \
		= left_right, left_UP, left_dir, cam_set[0]
		left_matrix[3,3] = 1
		self.left_matrix = np.linalg.inv(left_matrix)[:3,:]  # left_to_world

		right_dir = cam_set[4] - cam_set[3]
		right_right = np.cross(right_dir,cam_set[5])
		right_UP = np.cross(right_dir, right_right)

		right_dir = right_dir / np.linalg.norm(right_dir)
		right_right = right_right / np.linalg.norm(right_right)
		right_UP = right_UP / np.linalg.norm(right_UP)

		right_matrix = np.zeros((4,4), dtype='float64')
		right_matrix[:3,0], right_matrix[:3,1], right_matrix[:3,2], right_matrix[:3,3]\
		 = right_right, right_UP, right_dir, cam_set[3]
		right_matrix[3,3] = 1

		self.right_matrix = np.linalg.inv(right_matrix)[:3,:] # right_to_world

	def make_heat_map(self, heat, u,v, num):
		
		for i in range(v-3,v+4):
			for j in range(u-3, u+4):
				dist = np.linalg.norm([v-i, u-j], axis=-1)
				heat[i,j,num]= int(255*(1-dist/sqrt(18)))

		return heat

	def empty_heat(self, heat):

		H,W,C = heat.shape

		empty = 255 - heat.sum(2)
		empty = empty.reshape(H,W,1)

		with_empty = np.concatenate((heat,empty),axis=2)

		return with_empty

	def get_uv(self, key_list):

		heat_map_l = np.zeros((256,256,self.KEY_NUM), dtype='float64')
		heat_map_r = np.zeros((256,256,self.KEY_NUM), dtype='float64')

		uv_l = np.zeros((self.KEY_NUM,2))
		uv_r = np.zeros((self.KEY_NUM,2))

		for i in range(self.KEY_NUM):	
			target_pos = np.concatenate((key_list[i], (1,))) # x,y,z,1

			left_coor = np.matmul(self.left_matrix, target_pos.T) # left to world * world vector
			left_uv = np.matmul(self.K,left_coor)
			left_uv = (1/left_uv[2])*left_uv

			right_coor = np.matmul(self.right_matrix, target_pos.T)
			right_uv = np.matmul(self.K,right_coor)
			right_uv = (1/right_uv[2])*right_uv

			u1 = int(left_uv[0])
			v1 = int(left_uv[1])

			u2 = int(right_uv[0])
			v2 = int(right_uv[1])

			heat_map_l = self.make_heat_map(heat_map_l, u1, v1, i)
			heat_map_r = self.make_heat_map(heat_map_r, u2, v2, i)

			uv_l[i,0] = v1
			uv_l[i,1] = u1
			uv_r[i,0] = v2
			uv_r[i,1] = u2

		heat_map_l = self.empty_heat(heat_map_l)
		heat_map_r = self.empty_heat(heat_map_r)

		return heat_map_l, heat_map_r, uv_l, uv_r

	def get_crop(self, left, right, seg_l, seg_r, dep_l, dep_r, index, heat_l, heat_r):

		margin = random.randint(10,15)

		mask_l = np.zeros((256,256))
		mask_r = np.zeros((256,256))
		for ind in index:
			mask_l[seg_l == ind] = 1
			mask_r[seg_r == ind] = 1

		l_x_min = min(np.where(mask_l==1)[1])
		l_x_max = max(np.where(mask_l==1)[1])
		l_x_mean = int((l_x_min + l_x_max)/2.0)
		l_x_dist = l_x_max - l_x_min

		l_y_min = min(np.where(mask_l==1)[0])
		l_y_max = max(np.where(mask_l==1)[0])
		l_y_mean = int((l_y_min + l_y_max)/2.0)
		l_y_dist = l_y_max - l_y_min

		r_x_min = min(np.where(mask_r==1)[1])
		r_x_max = max(np.where(mask_r==1)[1])
		r_x_mean = int((r_x_min + r_x_max)/2.0)
		r_x_dist = r_x_max - r_x_min

		r_y_min = min(np.where(mask_r==1)[0])
		r_y_max = max(np.where(mask_r==1)[0])
		r_y_mean = int((r_y_min + r_y_max)/2.0)
		r_y_dist = r_y_max - r_y_min

		dist_x = max(l_x_dist,r_x_dist)
		dist_y = max(l_y_dist,r_y_dist)

		crop_left = left[l_y_mean - dist_y//2 - margin:l_y_mean + dist_y//2 + margin, l_x_mean - dist_x//2 - margin: l_x_mean + dist_x//2 + margin, :]
		crop_right = right[r_y_mean - dist_y//2 - margin:r_y_mean + dist_y//2 + margin, r_x_mean - dist_x//2 - margin: r_x_mean + dist_x//2 + margin, :]

		seg_left = mask_l[l_y_mean - dist_y//2 - margin:l_y_mean + dist_y//2 + margin, l_x_mean - dist_x//2 - margin: l_x_mean + dist_x//2 + margin]
		seg_right = mask_r[r_y_mean - dist_y//2 - margin:r_y_mean + dist_y//2 + margin, r_x_mean - dist_x//2 - margin: r_x_mean + dist_x//2 + margin]

		seg_left = np.expand_dims(seg_left,axis=2)
		seg_right = np.expand_dims(seg_right,axis=2)
		
		dep_l = dep_l.reshape((256,256,1))
		dep_r = dep_r.reshape((256,256,1))

		crop_depth_left = dep_l[l_y_mean - dist_y//2 - margin:l_y_mean + dist_y//2 + margin, l_x_mean - dist_x//2 - margin: l_x_mean + dist_x//2 + margin, :]
		crop_depth_right = dep_r[r_y_mean - dist_y//2 - margin:r_y_mean + dist_y//2 + margin, r_x_mean - dist_x//2 - margin: r_x_mean + dist_x//2 + margin, :]

		# min_dep = min(crop_depth_left.min(), crop_depth_right.min())
		# max_dep = max(crop_depth_left.max(), crop_depth_right.max())

		min_l_dep, min_r_dep = crop_depth_left.min(),crop_depth_right.min()	 
		max_l_dep, max_r_dep = crop_depth_left.max(), crop_depth_right.max()

		crop_depth_left = (crop_depth_left-min_l_dep)/(max_l_dep - min_l_dep)
		crop_depth_right = (crop_depth_right-min_r_dep)/(max_r_dep - min_r_dep)

		# crop_depth_left = (crop_depth_left-min_dep)/(max_dep - min_dep)
		# crop_depth_right = (crop_depth_right-min_dep)/(max_dep - min_dep)

		image_height = dist_y + 2*margin
		image_width  = dist_x + 2*margin

		crop_heat_l = heat_l[l_y_mean - dist_y//2 - margin:l_y_mean + dist_y//2 + margin, l_x_mean - dist_x//2 - margin: l_x_mean + dist_x//2 + margin, :]
		crop_heat_r = heat_r[r_y_mean - dist_y//2 - margin:r_y_mean + dist_y//2 + margin, r_x_mean - dist_x//2 - margin: r_x_mean + dist_x//2 + margin, :]


		return crop_left, crop_right, crop_depth_left, crop_depth_right, np.array([l_y_mean, l_x_mean]), np.array([r_y_mean, r_x_mean]), np.array([image_height, image_width]), crop_heat_l, crop_heat_r, seg_left, seg_right
