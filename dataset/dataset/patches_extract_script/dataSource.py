
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
#import natsort
from abc import ABC, abstractmethod

class DataSource(object):
	def __init__(self, band_n, foldernameInput, label_folder,name):
		self.band_n = band_n
		self.foldernameInput = foldernameInput
		self.label_folder = label_folder
		self.name=name

	
	@abstractmethod
	def im_load(self,filename):
		pass

class SARSource(DataSource):

	def __init__(self):
		name='SARSource'
		band_n = 2
		foldernameInput = "in_np2/"
		label_folder = 'labels'
		super().__init__(band_n, foldernameInput, label_folder,name)

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
	def im_load(self,filename):
		return np.load(filename)
class OpticalSource(DataSource):
	
	def __init__(self):
		name='OpticalSource'
		band_n = 2
		#self.t_len = self.dataset.getT_len() implement dataset classes here. then select the dataset/source class
		foldernameInput = "in_optical/"
		label_folder = 'optical_labels'
		# to-do: add input im list names: in_filenames=['01_aesffes.tif', '02_fajief.tif',...]
		super().__init__(band_n, foldernameInput, label_folder,name)

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
	def im_load(self,filename):
		return np.load(filename)[3,1,0] #3,1,0 means nir,g,b.
class Dataset(object):
	def __init__(self,path,im_h,im_w,class_n):
		self.path=path
		self.class_n=class_n
		self.im_h=im_h
		self.im_w=im_w
	@abstractmethod
	def addDataSource(self,dataSource):
		pass
class CampoVerde(Dataset):
	def __init__(self):
		path="../cv_data/"
		class_n=13
		im_h=8492
		im_w=7995
		super().__init__(path,im_h,im_w,class_n)

	def addDataSource(self,dataSource):
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			self.im_list=['20151029_S1', '20151110_S1', '20151122_S1', '20151204_S1', '20151216_S1', '20160121_S1', '20160214_S1', '20160309_S1', '20160321_S1', '20160508_S1', '20160520_S1', '20160613_S1', '20160707_S1', '20160731_S1']
		elif self.dataSource.name == 'OpticalSource':
			self.im_list=[]
		self.t_len=len(self.im_list)
class LEM(Dataset):
	def __init__(self):
		path="../lm_data/"
		class_n=15
		im_w=8658
		im_h=8484
		super().__init__(path,im_h,im_w,class_n)

	def addDataSource(self,dataSource):
		self.dataSource = dataSource
		if self.dataSource.name == 'SARSource':
			self.im_list=['20170612_S1', '20170706_S1', '20170811_S1', '20170916_S1', '20171010_S1', '20171115_S1', '20171209_S1', '20180114_S1', '20180219_S1', '20180315_S1', '20180420_S1', '20180514_S1', '20180619_S1']
		elif self.dataSource.name == 'OpticalSource':
			self.im_list=['20170729_S2_10m','20170803_S2_10m','20170907_S2_10m','20171017_S2_10m','20171022_S2_10m','20180420_S2_10m','20180430_S2_10m','20180510_S2_10m','20180614_S2_10m','20180619_S2_10m','20180624_S2_10m']
		self.t_len=len(self.im_list)
