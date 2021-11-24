from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from train_classification import train_classification
from generate import generate
# from generate_show_possibility_map import generate
import scipy.ndimage
import time

IS_TRAINING = False #True #

BATCH_SIZE = 32
EPOCHES = 5
LOGGING = 50
MODEL_SAVE_PATH = './models_EC/4/4.ckpt'

def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		f = h5py.File('vis_ir128.h5', 'r')
		sources = f['data'][:]

		sources = np.transpose(sources, (0, 3, 2, 1))
		print("sources shape:", sources.shape)
		sources = sources / 255.0
		train_classification(sources, './models_EC/', EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		ir = './test_imgs/ir'
		vi = './test_imgs/vis'
		fused_path = './results/'
		filelist = os.listdir(ir)
		t = []

		N = 0
		for item in filelist:
			if item.endswith('.bmp') or item.endswith('.png') or item.endswith('.jpg'):
				N=N+1
				ir_path = os.path.join(ir, item)
				vis_path = os.path.join(vi, item)
				savepath = os.path.join(fused_path, item)
				print("\033[0;33;40m")
				print("process: [%d/%d]" % (N, len(filelist)))
				print(ir_path)
				print(vis_path)
				print("\033[0m")

				begin = time.time()
				generate(ir_path, vis_path, MODEL_SAVE_PATH, model_num = 4, output_path = savepath)
				end = time.time()
				t.append(end - begin)
		print("mean:%s, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()
