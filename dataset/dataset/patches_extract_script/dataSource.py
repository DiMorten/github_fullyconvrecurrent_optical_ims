
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
#import json
import random
#import pprint
#import scipy.misc
import numpy as np
from time import gmtime, strftime
#from osgeo import gdal
import glob
#from skimage.transform import resize
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
import cv2
import pathlib
#from sklearn.feature_extraction.image import extract_patches_2d
#from skimage.util import view_as_windows
import sys
import pickle
# Local
import deb
import argparse
from sklearn.preprocessing import StandardScaler
from skimage.util import view_as_windows

from abc import ABC, abstractmethod

class DataSource(object):
	def __init__(self, band_n, foldernameInput, label_folder):
		self.band_n = band_n
		self.foldernameInput = foldernameInput
		self.label_folder = label_folder

	
	@abstractmethod
	def im_seq_normalize3(self,im,mask):
		pass

class SARSource(DataSource):

	def __init__(self):
		band_n = 2
		foldernameInput = "in_np2/"
		label_folder = 'labels'
		super().__init__(band_n, foldernameInput, label_folder)

	def im_seq_normalize3(self,im,mask):
		
		t_steps,h,w,channels=im.shape
		#im=im.copy()
		im_flat=np.transpose(im,(1,2,3,0))
		#im=np.reshape(im,(h,w,t_steps*channels))
		im_flat=np.reshape(im_flat,(h*w,channels*t_steps))
		im_check=np.reshape(im_flat,(h,w,channels,t_steps))
		im_check=np.transpose(im_check,(3,0,1,2))

		deb.prints(im_check.shape)
		deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		mask_flat=np.reshape(mask,-1)
		train_flat=im_flat[mask_flat==1,:]

		deb.prints(train_flat.shape)
		print(np.min(train_flat),np.max(train_flat),np.average(train_flat))

		scaler=StandardScaler()
		scaler.fit(train_flat)
		train_norm_flat=scaler.transform(train_flat)

		im_norm_flat=scaler.transform(im_flat)
		im_norm=np.reshape(im_norm_flat,(h,w,channels,t_steps))
		deb.prints(im_norm.shape)
		im_norm=np.transpose(im_norm,(3,0,1,2))
		deb.prints(im_norm.shape)
		#for t_step in range(t_steps):
		#	print("Normalized time",t_step)
		#	print(np.min(im_norm[t_step]),np.max(im_norm[t_step]),np.average(im_norm[t_step]))
		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		return im_norm
	def clip_undesired_values(self, full_ims):
		full_ims[full_ims>1]=1
		return full_ims

class OpticalSource(DataSource):
	
	def __init__(self):
		band_n = 2
		#self.t_len = self.dataset.getT_len() implement dataset classes here. then select the dataset/source class
		foldernameInput = "in_optical/"
		label_folder = 'optical_labels'
		# to-do: add input im list names: in_filenames=['01_aesffes.tif', '02_fajief.tif',...]
		super().__init__(band_n, foldernameInput, label_folder)

	def im_seq_normalize3(self,im,mask): #to-do: check if this still works for optical
		
		t_steps,h,w,channels=im.shape
		#im=im.copy()
		im_flat=np.transpose(im,(1,2,3,0))
		#im=np.reshape(im,(h,w,t_steps*channels))
		im_flat=np.reshape(im_flat,(h*w,channels*t_steps))
		im_check=np.reshape(im_flat,(h,w,channels,t_steps))
		im_check=np.transpose(im_check,(3,0,1,2))

		deb.prints(im_check.shape)
		deb.prints(np.all(im_check==im))
		deb.prints(im.shape)
		mask_flat=np.reshape(mask,-1)
		train_flat=im_flat[mask_flat==1,:]

		deb.prints(train_flat.shape)
		print(np.min(train_flat),np.max(train_flat),np.average(train_flat))

		scaler=StandardScaler()
		scaler.fit(train_flat)
		train_norm_flat=scaler.transform(train_flat)

		im_norm_flat=scaler.transform(im_flat)
		im_norm=np.reshape(im_norm_flat,(h,w,channels,t_steps))
		deb.prints(im_norm.shape)
		im_norm=np.transpose(im_norm,(3,0,1,2))
		deb.prints(im_norm.shape)
		#for t_step in range(t_steps):
		#	print("Normalized time",t_step)
		#	print(np.min(im_norm[t_step]),np.max(im_norm[t_step]),np.average(im_norm[t_step]))
		print("FINISHED NORMALIZING, RESULT:")
		print(np.min(im_norm),np.max(im_norm),np.average(im_norm))
		return im_norm
	def clip_undesired_values(self, full_ims):
		#full_ims[full_ims]>7000]=1
		return full_ims