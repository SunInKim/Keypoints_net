#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi

import roslib
import rospy
from sensor_msgs.msg import Image

class CHECK(QDialog):

	def __init__(self):
		super(CHECK,self).__init__()
		loadUi('3D_check_ui.ui',self)

		self.setWindowTitle('3D_CHECK')

		self.bridge = CvBridge()

		self.real_x, self.real_y, self.real_z = 0.4, 0.0, -0.01
		self.u1, self.v1, self.u2, self.v2 = 0,0,0,0
		self.key_list = []
		self.uv_list = []
		self.save_count = 0

		self.params_path = '/home/irobot/catkin_ws/src/yolov3/camera_params'
		self.save_path = 	'test_data'

		self.left_topic = '/kinectB/rgb/image_raw'
		self.right_topic = '/kinectC/rgb/image_raw'

		rospy.Subscriber(self.left_topic, Image, self.callback_left)
		rospy.Subscriber(self.right_topic, Image, self.callback_right)

		############ Button Connection ############
		self.Slider_X.valueChanged.connect(self.change_X)	
		self.Slider_Y.valueChanged.connect(self.change_Y)	
		self.Slider_Z.valueChanged.connect(self.change_Z)
		self.Start_Btn.clicked.connect(self.start)	
		self.Reset_Btn.clicked.connect(self.reset)
		self.Load_Btn.clicked.connect(self.load)
		self.Save_Btn.clicked.connect(self.save)
		self.Key_Save_Btn.clicked.connect(self.key_save)

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
		
	def change_X(self):

		self.real_x = self.Slider_X.value()/1000
		self.Status_x.setText("%i"%self.Slider_X.value())
		self.cal_uv()

	def change_Y(self):

		self.real_y = self.Slider_Y.value()/1000
		self.Status_y.setText("%i"%self.Slider_Y.value())
		self.cal_uv()


	def change_Z(self):

		self.real_z = self.Slider_Z.value()/1000
		self.Status_z.setText("%i"%self.Slider_Z.value())
		self.cal_uv()

	def cal_uv(self):

		position = np.array([[self.real_x, self.real_y, self.real_z, 1]])

		left_coor = np.matmul(self.h_matrix_B, position.T) # left to world * world vector
		left_uv = np.matmul(self.cam_B_K,left_coor[:3,:])
		left_uv = (1/left_uv[2])*left_uv

		right_coor = np.matmul(self.h_matrix_C, position.T)
		right_uv = np.matmul(self.cam_C_K,right_coor[:3,:])
		right_uv = (1/right_uv[2])*right_uv

		self.u1 = int(256*left_uv[0]/1280.0)
		self.v1 = int(256*left_uv[1]/720.0)

		self.u2 = int(256*right_uv[0]/1280.0)
		self.v2 = int(256*right_uv[1]/720.0)

		print("Position X :", self.real_x,"Y :", self.real_y, "Z :", self.real_z)


	def draw_dot(self):
		left_img = self.left_rgb.copy()
		right_img = self.right_rgb.copy()

		left_img[self.v1,:,:] = 0
		left_img[:,self.u1,:] = 0
		left_img = cv2.circle(left_img,(self.u1,self.v1),3,(255,0,0),-1)
		right_img[self.v2,:,:] = 0
		right_img[:,self.u2,:] = 0
		right_img = cv2.circle(right_img,(self.u2,self.v2),3,(255,0,0),-1)

		return left_img, right_img

	def start(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.streaming_start)
		self.timer.start(1000/10)

	def streaming_start(self):

		left_frame, right_frame = self.draw_dot()

		self.left_qt_img = QtGui.QImage(left_frame, left_frame.shape[1],left_frame.shape[0], QImage.Format_RGB888)
		self.right_qt_img = QtGui.QImage(right_frame, right_frame.shape[1],right_frame.shape[0], QImage.Format_RGB888)
		self.left_pixmapImage = QtGui.QPixmap.fromImage(self.left_qt_img)		
		self.right_pixmapImage = QtGui.QPixmap.fromImage(self.right_qt_img)
		self.Left_video.setPixmap(self.left_pixmapImage)
		self.Right_video.setPixmap(self.right_pixmapImage)

	def save(self):		
		
		left_frame, right_frame = self.draw_dot()

		# plt.imsave(self.save_path + "/left/left_{}.png".format(self.save_count),left_frame)
		# plt.imsave(self.save_path + "/right/right_{}.png".format(self.save_count),right_frame)
		
		np.save(self.save_path+"/key/key_{}.npy".format(self.save_count),np.array(self.key_list))
		np.save(self.save_path+"/uv/uv_{}.npy".format(self.save_count),np.array(self.uv_list))
		self.save_count += 1

	def key_save(self):
		self.key_list.append([self.real_x,self.real_y,self.real_z])
		self.uv_list.append([self.v1,self.u1,self.v2,self.u2])
		print("saved")

	def reset(self):
		self.real_x, self.real_y, self.real_z = 0.4, 0.0, -0.01
		self.u1, self.v1, self.u2, self.v2 = 0, 0, 0, 0

		self.key_list = []
		self.uv_list = []



def main(args):
	rospy.init_node('CHECK', anonymous=True)

if __name__=='__main__':
	main(sys.argv)
	app = QApplication(sys.argv)
	widget = CHECK()
	widget.show()
	sys.exit(app.exec_())
