#!/usr/bin/env python3
import sys

NET_WORK = 'Mug'
epoch = 21

sys.path.append(NET_WORK)
import matplotlib.pyplot as plt

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import random

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi

import roslib
import rospy
from sensor_msgs.msg import Image

from models import *
from camera_cal import *

import torch
from torchvision.utils import save_image

class Two_Img(QDialog):

	def __init__(self):
		super(Two_Img,self).__init__()
		loadUi('Get2img.ui',self)

		self.setWindowTitle('Two Img')

		self.bridge = CvBridge()

		self.keypoint_net = KeypointNet(key_num=3)
		self.keypoint_net = self.keypoint_net.cuda()
		self.keypoint_net.load_state_dict(torch.load("{}/keypoint_net_{}.pth".format(NET_WORK,epoch)))


		self.real_x, self.real_y, self.real_z = 0, 0, 0
		self.u1, self.v1, self.u2, self.v2 = 0,0,0,0

		self.left_clicked, self.right_clicked = False, False
		self.left_pre, self.right_pre = np.zeros((256,256,3)), np.zeros((256,256,3))
		self.save_count = 0

		self.params_path = '/home/irobot/catkin_ws/src/yolov3/camera_params'
		self.save_path = 'test_data'

		self.left_topic = '/kinectB/rgb/image_raw'
		self.right_topic = '/kinectC/rgb/image_raw'

		rospy.Subscriber(self.left_topic, Image, self.callback_left)
		rospy.Subscriber(self.right_topic, Image, self.callback_right)

		############ Button Connection ############
		self.Slider_u1.valueChanged.connect(self.change_u1)	
		self.Slider_v1.valueChanged.connect(self.change_v1)	
		self.Slider_u2.valueChanged.connect(self.change_u2)	
		self.Slider_v2.valueChanged.connect(self.change_v2)
		self.Start_Btn.clicked.connect(self.start)	
		self.Cal_Btn.clicked.connect(self.calculation)
		self.Load_Btn.clicked.connect(self.load)
		self.Save_Btn.clicked.connect(self.save)

		self.Left_Btn.clicked.connect(self.left_bt)
		self.Right_Btn.clicked.connect(self.right_bt)

	def left_bt(self):

		self.left_clicked = True
		self.right_clicked = False

	def right_bt(self):

		self.right_clicked = True
		self.left_clicked = False

	def callback_left(self, data):
		try:
			self.left_rgb = self.bridge.imgmsg_to_cv2(data,"rgb8")
			self.left_rgb = cv2.resize(self.left_rgb,(256, 256))	
			self.get_left = True
		except CvBridgeError as e:
			print(e)

	def callback_right(self, data):
		try:
			self.right_rgb = self.bridge.imgmsg_to_cv2(data,"rgb8")
			self.right_rgb = cv2.resize(self.right_rgb,(256, 256))
			self.get_right = True
		except CvBridgeError as e:
			print(e)


	def load(self):

		self.cam_B_K = np.load(self.params_path + '/camB_K.npy')
		self.cam_C_K = np.load(self.params_path + '/camC_K.npy')
		self.h_matrix_B = np.load(self.params_path + '/l2w.npy')
		self.h_matrix_C = np.load(self.params_path + '/r2w.npy')

		rospy.loginfo("Camera parameters received")
		
	def change_u1(self):

		self.u1 = self.Slider_u1.value()
		self.Status_u1.setText("%i"%self.Slider_u1.value())
		self.draw_dot()

	def change_v1(self):

		self.v1 = self.Slider_v1.value()
		self.Status_v1.setText("%i"%self.Slider_v1.value())
		self.draw_dot()


	def change_u2(self):

		self.u2 = self.Slider_u2.value()
		self.Status_u2.setText("%i"%self.Slider_u2.value())
		self.draw_dot()

	def change_v2(self):

		self.v2 = self.Slider_v2.value()
		self.Status_v2.setText("%i"%self.Slider_v2.value())
		self.draw_dot()

	def draw_dot(self):
		left_img = self.left_rgb.copy()
		right_img = self.right_rgb.copy()

		if self.left_clicked:
			left_img[self.v1,:,:] = 0
			left_img[:,self.u1,:] = 0
			left_img = cv2.circle(left_img,(self.u1,self.v1),3,(255,0,0),-1)
			left_img[self.v2,:,:] = 0
			left_img[:,self.u2,:] = 0
			left_img = cv2.circle(left_img,(self.u2,self.v2),3,(255,0,0),-1)
			self.left_pre = left_img
			self.left_detect = [self.v1, self.u1, self.v2, self.u2]

		elif self.right_clicked:
			right_img[self.v1,:,:] = 0
			right_img[:,self.u1,:] = 0
			right_img = cv2.circle(right_img,(self.u1,self.v1),3,(255,0,0),-1)
			right_img[self.v2,:,:] = 0
			right_img[:,self.u2,:] = 0
			right_img = cv2.circle(right_img,(self.u2,self.v2),3,(255,0,0),-1)
			self.right_pre = right_img
			self.right_detect = [self.v1, self.u1, self.v2, self.u2]

	def start(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.streaming_start)
		self.timer.start(1000/10)

	def streaming_start(self):

		left_frame, right_frame = self.left_pre, self.right_pre

		self.left_qt_img = QtGui.QImage(left_frame, left_frame.shape[1],left_frame.shape[0], QImage.Format_RGB888)
		self.right_qt_img = QtGui.QImage(right_frame, right_frame.shape[1],right_frame.shape[0], QImage.Format_RGB888)
		self.left_pixmapImage = QtGui.QPixmap.fromImage(self.left_qt_img)		
		self.right_pixmapImage = QtGui.QPixmap.fromImage(self.right_qt_img)
		self.Left_video.setPixmap(self.left_pixmapImage)
		self.Right_video.setPixmap(self.right_pixmapImage)

	def save(self):
		left_img = self.left_rgb.copy()
		right_img = self.right_rgb.copy()

		l_v1,l_u1,l_v2,l_u2 = self.left_detect[0], self.left_detect[1], self.left_detect[2], self.left_detect[3]
		left_h = l_v2-l_v1
		left_w = l_u2-l_u1
		left_m_h = (l_v2+l_v1)//2
		left_m_w = (l_u2+l_u1)//2

		r_v1,r_u1,r_v2,r_u2 = self.right_detect[0], self.right_detect[1], self.right_detect[2], self.right_detect[3]
		right_h = r_v2-r_v1
		right_w = r_u2-r_u1
		right_m_h = (r_v2+r_v1)//2
		right_m_w = (r_u2+r_u1)//2

		img_h = max(left_h,right_h)
		img_w = max(left_w,right_w)

		img_h = img_h + img_h%2
		img_w = img_w + img_w%2

		margin = random.randint(10,15)
		margin = 5

		self.crop_left = left_img[left_m_h-img_h//2-margin:left_m_h+img_h//2+margin,left_m_w-img_w//2-margin:left_m_w+img_w//2+margin,:]
		self.crop_right = right_img[right_m_h-img_h//2-margin:right_m_h+img_h//2+margin,right_m_w-img_w//2-margin:right_m_w+img_w//2+margin,:]

		# con_img = np.concatenate((self.crop_left,self.crop_right),axis=1)
		# plt.imsave('{}/images/left_{}.png'.format(NET_WORK,self.save_count),self.crop_left)
		# plt.imsave('{}/images/right_{}.png'.format(NET_WORK,self.save_count),self.crop_right)

		plt.imsave(self.save_path + '/left/left_{}.png'.format(self.save_count),self.crop_left)
		plt.imsave(self.save_path + '/right/right_{}.png'.format(self.save_count),self.crop_right)

	def calculation(self):
		
		# Visualize feature maps
		activation = {}
		def get_activation(name):
		    def hook(model, input, output):
		        activation[name] = output.detach()
		    return hook

		self.keypoint_net.feature_left.up8.register_forward_hook(get_activation('up8'))

		
		left_input, right_input = np.transpose(self.crop_left.copy(),[2, 0, 1]), np.transpose(self.crop_right.copy(),[2, 0, 1])
		left_input, right_input = torch.FloatTensor(left_input)/255.0, torch.FloatTensor(right_input)/255.0
		left_input, right_input = left_input.unsqueeze(0).cuda(), right_input.unsqueeze(0).cuda()

		# _, rgb_l, depth_l, final_heat_l, _, rgb_r, depth_r, final_heat_r, feature_map = self.keypoint_net(left_input, right_input,mode="val")
		
		# middle_heat_l, rgb_l, depth_l, final_heat_l, middle_heat_r, rgb_r, depth_r, final_heat_r = self.keypoint_net(left_input, right_input,mode="val") 
		middle_heat_l, final_heat_l, middle_heat_r, final_heat_r, rgb, depth= self.keypoint_net(left_input, right_input,mode="val") 

		act = activation['up8'].squeeze()
		fig, axarr = plt.subplots(16,16)
		for idx in range(16):
			for jj in range(16):
			    axarr[idx,jj].imshow(act[16*idx+jj].cpu())
		#fig.axis('off')
		plt.savefig("{}/images/featuremap_{}.png".format(NET_WORK,self.save_count),bbox_inches='tight',pad_inches=0)

		# f_dep = torch.cat((depth_l,depth_r),dim=3)
		# f_dep = torch.cat((f_dep,f_dep,f_dep),dim=1)

		f_dep = torch.cat((depth[:,:1,:,:],depth[:,1:,:,:]),dim=3)
		f_dep = torch.cat((f_dep,f_dep,f_dep),dim=1)

		in_visu = torch.cat((left_input, right_input),dim=3)
		gen_visu_heat1 = torch.cat((final_heat_l[:,:3,:,:],final_heat_r[:,:3,:,:]),dim=3)
		# gen_visu_heat2 = torch.cat((final_heat_l[:,2:5,:,:],final_heat_r[:,2:5,:,:]),dim=3)

		img_sample = torch.cat((in_visu.data, f_dep.data, gen_visu_heat1.data), -2)

		final_heat_l, final_heat_r = final_heat_l.detach().cpu().squeeze(0), final_heat_r.detach().cpu().squeeze(0)
		final_heat_l, final_heat_r = np.array(final_heat_l), np.array(final_heat_r)
		final_heat_l, final_heat_r = final_heat_l[:3,:,:], final_heat_r[:3,:,:]

		left_uv_list = cal_uv(final_heat_l,3)
		right_uv_list = cal_uv(final_heat_r,3)

		left_crop = draw_heat(self.crop_left.copy(),left_uv_list,3)
		right_crop = draw_heat(self.crop_right.copy(),right_uv_list,3)

		plt.imsave("{}/images/left_crop_{}.png".format(NET_WORK,self.save_count),left_crop/255.0)
		plt.imsave("{}/images/right_crop_{}.png".format(NET_WORK,self.save_count),right_crop/255.0)


		save_image(img_sample, "{}/images/detect_{}.png".format(NET_WORK,self.save_count), nrow=5, normalize=True)
		self.save_count += 1

def main(args):
	rospy.init_node('Two_Img', anonymous=True)

if __name__=='__main__':
	main(sys.argv)
	app = QApplication(sys.argv)
	widget = Two_Img()
	widget.show()
	sys.exit(app.exec_())
