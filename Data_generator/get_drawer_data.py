import os
import time
import pdb
import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
from math import pi
import matplotlib.pyplot as plt
import random
from camera_cal import *

robot_index = []
table_index = []
wall_index = []
laptop_index = []
drawer_index = []
obj_index = []

def loadTexture():
	texture_path = "texture/"
	texture_list = os.listdir(texture_path)
	textures =[]
	for tx in texture_list:
		tx_id = p.loadTexture(texture_path + tx)
		textures.append(tx_id)


	return textures

save_num = 2000
save_count = 0
seed1 = 45
serverMode = p.GUI # GUI/DIRECT
physicsClient = p.connect(serverMode)

for drawer_num in range(1,6):
	# drawer_num = int(input("Drawer_num?"))
	# scale = float(input("Scaling?"))
	# save_count = save_num*(drawer_num-1)
	# save_count = int(input("Save start count?"))
	# seed1 = int(input("seed?"))
	save_path = "data/"
	random.seed(seed1)

	sisbotUrdfPath = "urdf.franka_panda/panda.urdf"

	scale = random.uniform(1.0,1.2)
	max_dist_list = scale*np.array([0.11, 0.11, 0.10, 0.12, 0.08])

	max_dist = max_dist_list[drawer_num-1]

	# connect to engine servers
	# add search path for loadURDFs
	p.setAdditionalSearchPath(pybullet_data.getDataPath())

	# define world
	p.setGravity(0,0,-9.8) # NOTE

	
	# Load wall
	wall1Orientation = p.getQuaternionFromEuler([0, 0, 1.57079632679])
	wall2Orientation = p.getQuaternionFromEuler([0, 0, 1.57079632679])
	wall1Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.584 + 0.3 + 0.025 , 0.0, 1.04-0.554], useFixedBase = True)
	wall2Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.3+ 0.584, -0.625, 1.04-0.554], wall1Orientation, useFixedBase = True)
	wall3Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.3+ 0.584,  0.625, 1.04-0.554], wall2Orientation, useFixedBase = True)

	# Load table
	tableStartPos = [0.584, 0.0, 0.524-0.554]
	tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
	tableID = p.loadURDF("./urdf/objects/table/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

	# Load drawer
	drawerStartPos = [0.68, 0.2, 0.545-0.554]
	drawerStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
	drawerId = p.loadURDF("./urdf/objects/drawer{}/drawer{}.urdf".format(drawer_num,drawer_num), drawerStartPos, drawerStartOrientation,globalScaling=scale,useFixedBase = True,flags=p.URDF_USE_SELF_COLLISION)
	p.setJointMotorControl2(drawerId, 0, p.VELOCITY_CONTROL, force=5)
	p.setJointMotorControl2(drawerId, 2, p.VELOCITY_CONTROL, force=5)

	# Setup panda with robotiq 85
	robotStartPos = [0,0,0.554-0.554]
	robotStartOrn = p.getQuaternionFromEuler([0,0,0])
	print("----------------------------------------")
	print("Loading robot from {}".format(sisbotUrdfPath))
	robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True)
	joints, controlJoints= utils_ur5_robotiq140.setup_sisbot(p, robotID)
	eefID = 11 # ee_link

	DRAWER_COLOR = [[1.0, 1.0, 1.0, 1],[0.9, 0.10, 0.68, 1], [0.937, 0.729, 0.494, 1],[0.0, 0.4196, 0.502, 1],[0.82, 0.82, 0.82, 1]]

	k_list = [10,11,5,8,3]
	link_list = [-1,0,1,2,4]

	KEY_NUM = len(k_list)
	cal_lib = camera_calculation(key_num=KEY_NUM)

	# Indexing
	for i in range(21):
		robot_index.append(robotID+((i)<<24))

	for i in range(2):
		table_index.append(tableID+((i)<<24))

	for i in range(2):
		wall_index.append(wall1Id+((i)<<24))
		wall_index.append(wall2Id+((i)<<24))
		wall_index.append(wall3Id+((i)<<24))

	for i in range(17):
		drawer_index.append([drawerId+(i-1<<24)])

	# Camera setting
	width = 256
	height = 256

	cam_pos1 = np.array([-0.027, 0.436, 0.939-0.554])
	cam_tar1 = np.array([1.288, -0.086, 0.468-0.554])
	cam_up1 = np.array([1.19, 0.00, 0.992])
	cam_pos2 = np.array([-0.027, -0.436, 0.939-0.554])
	cam_tar2 = np.array([1.288, 0.086, 0.468-0.554])
	cam_up2 = np.array([1.19, 0.00, 0.992])

	cam_set = [cam_pos1, cam_tar1, cam_up1, cam_pos2, cam_tar2, cam_up2]

	view_matrix  = p.computeViewMatrix(cam_pos1, cam_tar1, cam_up1)
	view_matrix2 = p.computeViewMatrix(cam_pos2, cam_tar2, cam_up2)
	fov = 59
	aspect = 1280.0/720.0
	near = 0.1
	far = 1.5
	projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
			
	userParams = dict()
	for j in range(save_num):
		cam_random = np.empty((6,3))
		for i in range(6):
			cam_rand = np.random.uniform(-0.07,0.07,3)
			cam_random[i,:] = cam_set[i] + cam_rand

		view_matrix  = p.computeViewMatrix(cam_random[0], cam_random[1], cam_random[2])
		view_matrix2 = p.computeViewMatrix(cam_random[3], cam_random[4], cam_random[5])

		color = np.random.uniform(0,1,4)
		cal_lib.cal_matrix(cam_random)

		while color[:3].mean() < 0.5:
			color = np.random.uniform(0,1,4)

		for i in link_list:
			add = np.random.uniform(-0.08,0.08,4)
			color += add
			color = np.clip(color,0,1)
			color[3] = 1.0
			p.changeVisualShape(drawerId,i, rgbaColor=color)

		color = np.random.uniform(0,1,4)
		color[3] = 1.0
		p.changeVisualShape(wall1Id,-1, rgbaColor=color)
		p.changeVisualShape(wall2Id,-1, rgbaColor=color)
		p.changeVisualShape(wall3Id,-1, rgbaColor=color)

		color = np.random.uniform(0,1,4)
		color[3] = 1.0
		p.changeVisualShape(tableID,-1, rgbaColor=color)

		userParams = [0, 0.1, 0, -1.5 ,0, 1.6, 0.7963762844]
		tt = 0

		draw_z1 = 0.545-0.554
		draw_x1 = random.uniform(0.58, 0.70)
		draw_y1 = random.uniform(-0.35, 0.35)
		if draw_y1 < 0:
			draw_yaw1 = random.uniform(3.141592/2.0 + 0.6  , 3.141592 +3.141592/2.0*(1+draw_y1/0.32))
		else: 
			draw_yaw1 = random.uniform(-3.141592 -3.141592/2.0*(1-draw_y1/0.32), -3.141592/2.0 - 0.6)
		drawer_pos = [draw_x1,draw_y1,draw_z1]
		drawer_ori = p.getQuaternionFromEuler([0,0,draw_yaw1])
		dist = random.uniform(0.00, max_dist)
		# dist = 0.07
		# drawer_pos = [draw_x1,0.30,draw_z1]

		# Robot joint
		for i, name in enumerate(controlJoints):
			joint = joints[name]
			if i != 7:
				pose1 = userParams[i]
			if i >6:
				break
			p.resetJointState(robotID, joint.id, targetValue=pose1, targetVelocity=0)
		
		p.resetJointState(drawerId, 2, targetValue= dist, targetVelocity=0)
		p.resetJointState(drawerId, 0, targetValue= 0.0, targetVelocity=0)
		p.resetBasePositionAndOrientation(drawerId, drawer_pos ,drawer_ori)

		# Image setting
		direction = np.array([random.uniform(-5.5,2.5),random.uniform(-2.0,2.0),random.uniform(1.5,7.5)])
		Color=np.random.uniform(0.75,0.99,3)
		AmbientCoeff=random.uniform(0.4,0.6)
		DiffuseCoeff=random.uniform(0.4,0.6)
		SpecularCoeff=random.uniform(0.01,0.03)

		# lightDirection=direction,	
		images = p.getCameraImage(width,height,view_matrix,projection_matrix, shadow=True, lightColor=Color, \
									lightAmbientCoeff=AmbientCoeff, lightDirection=direction, lightDiffuseCoeff=DiffuseCoeff, \
									lightSpecularCoeff=SpecularCoeff)       
		rgb_opengl = np.reshape(images[2], (height, width, 4))
		dep_l = images[3]
		seg_l = images[4]
		rgb_l = rgb_opengl[:,:,:3]
		
		images2 = p.getCameraImage(width,height,view_matrix2,projection_matrix,shadow=True, lightColor=Color, \
									lightAmbientCoeff=AmbientCoeff, lightDirection=direction, lightDiffuseCoeff=DiffuseCoeff, ]
									lightSpecularCoeff=SpecularCoeff)
		rgb_opengl = np.reshape(images2[2], (height, width, 4))
		dep_r = images2[3]
		seg_r = images2[4]
		rgb_r = rgb_opengl[:,:,:3]
		
		####################### SAVE ##########################

		keypoint_list = np.zeros((KEY_NUM,3))
		for i in range(KEY_NUM):
			k_position = np.array(p.getLinkState(drawerId, k_list[i])[0])
			for j in range(3):
				keypoint_list[i,j] = k_position[j]


		heat_map_l, heat_map_r, uv_l, uv_r = cal_lib.get_uv(keypoint_list)

		left_crop, right_crop, depth_left, depth_right, left_zero, right_zero, image_size, crop_heat_l, crop_heat_r, seg_left, seg_right = cal_lib.get_crop(rgb_l, rgb_r, seg_l, seg_r, dep_l, dep_r, drawer_index, heat_map_l, heat_map_r) 

		np.save(save_path+"left/depth/left_{}.npy".format(save_count), depth_left)
		np.save(save_path+"right/depth/right_{}.npy".format(save_count), depth_right)

		np.save(save_path+"left/crop/left_{}.npy".format(save_count), left_crop)
		np.save(save_path+"right/crop/right_{}.npy".format(save_count), right_crop)

		np.save(save_path+"left/seg/left_{}.npy".format(save_count), seg_left)
		np.save(save_path+"right/seg/right_{}.npy".format(save_count), seg_right)

		np.save(save_path+"left/uv/left_{}.npy".format(save_count), uv_l)
		np.save(save_path+"right/uv/right_{}.npy".format(save_count), uv_r)
		
		left_img_size = np.concatenate((left_zero ,image_size))
		right_img_size = np.concatenate((right_zero ,image_size))

		np.save(save_path+"left/zero_size/left_{}.npy".format(save_count), left_img_size)
		np.save(save_path+"right/zero_size/right_{}.npy".format(save_count), right_img_size)

		np.save(save_path+"left/heat/left_{}.npy".format(save_count),crop_heat_l)
		np.save(save_path+"right/heat/right_{}.npy".format(save_count),crop_heat_r)

		np.save(save_path+"keypoint/keypoint_{}.npy".format(save_count), keypoint_list)
		np.save(save_path+"camset/camset_{}.npy".format(save_count),cam_random)
		

		################## VISUAL SAVE ######################
		# l_visu = np.concatenate((depth_left,depth_left,depth_left), axis=2)
		# r_visu = np.concatenate((depth_right,depth_right,depth_right), axis=2)
		# visu = np.concatenate((l_visu,r_visu),axis =1)

		# visu2 = np.concatenate((crop_heat_l.sum(axis=2),crop_heat_r.sum(axis=2)),axis =1)

		# plt.imsave(save_path+"left/test/left_{}.png".format(save_count), visu2)
		# plt.imsave(save_path+"left/test/left_{}.png".format(save_count), heat_map_r.sum(axis=2))
		# np.savez_compressed(save_path+"left/image/left_{}.npz".format(save_count), rgb_l)
		# np.savez_compressed(save_path+"right/image/right_{}.npz".format(save_count), rgb_r)

		print(save_count)
		save_count+=1
		p.stepSimulation()

	p.resetSimulation()
	print("Next drawer num : ", save_count//2000 +1)
	print("Next start num : ", save_count) 
	print("Seed num : ", seed1)