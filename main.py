from __future__ import print_function

# from utils import list_images
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from train_classification import train_classification
import scipy.ndimage
import time
import tensorflow as tf
from networks import Encoder, Classification, Decoder
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import scipy.misc
import time
import scipy.io as scio
from ops import *
from integrated_gradients import IntegratedGradients

IS_TRAINING = False #True #

BATCH_SIZE = 32
EPOCHES = 5
LOGGING = 50


def main():
	if IS_TRAINING:
		f = h5py.File('vis_ir128.h5', 'r')
		sources = f['data'][:]

		sources = np.transpose(sources, (0, 3, 2, 1))
		print("sources shape:", sources.shape)
		sources = sources / 255.0
		print(('\nBegin to train the network ...\n'))
		train_classification(sources, './models_EC/', EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		path='test_imgs/'

		model_path = './models_EC/4/4.ckpt'

		t = []

		with tf.Graph().as_default() as graph:
			with tf.Session() as sess:
				SOURCE = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='SOURCE')
				FEAS = tf.placeholder(tf.float32, shape=(None, None, None, 24), name='FEATURES')

				Enco = Encoder('Encoder')
				Deco = Decoder('Decoder')
				feas = Enco.encode(image=SOURCE, is_training=False)
				RESULT = Deco.decode(features=FEAS, is_training=False)

				Class = Classification('Classification')
				out = Class.classification(image=FEAS, is_training=False)
				prob = tf.nn.softmax(out)

				sess.run(tf.global_variables_initializer())

				# restore the trained model
				theta_e = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Encoder')
				theta_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder')
				savered = tf.train.Saver(var_list=theta_e + theta_d)

				ED_model_num = str(4)
				savered.restore(sess, './models_ED/4/4.ckpt')

				theta_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Classification')
				saverc = tf.train.Saver(var_list=theta_c)
				saverc.restore(sess, model_path)


				for i in range(44):
					index = i + 1
					savepath = './results/' + str(index)
					ir_path = path + 'ir/' + str(index) + '.jpg'
					vis_path = path + 'vis/' + str(index) + '.jpg'
					print("\033[0;33;40m\t")
					print(ir_path)
					print(vis_path)
					print("\033[0m\t")

					begin = time.time()

					ir_img = (imread(ir_path) / 255.0 - 0.5) * 2
					vis_img = (imread(vis_path) / 255.0 - 0.5) * 2

					H, W = vis_img.shape

					ir_dimension = list(ir_img.shape)
					vis_dimension = list(vis_img.shape)
					ir_dimension.insert(0, 1)
					ir_dimension.append(1)
					vis_dimension.insert(0, 1)
					vis_dimension.append(1)
					ir_img = ir_img.reshape(ir_dimension)
					vis_img = vis_img.reshape(vis_dimension)

					FEED_DICT_VIS = {SOURCE: vis_img}
					FEED_DICT_IR = {SOURCE: ir_img}
					vis_feas_output = sess.run(feas, feed_dict=FEED_DICT_VIS)
					ir_feas_output = sess.run(feas, feed_dict=FEED_DICT_IR)

					pro_vis = sess.run(prob, feed_dict={
						FEAS: sess.run(feas, feed_dict={SOURCE: vis_img - np.mean(vis_img) + np.mean(ir_img)})})
					pro_ir = sess.run(prob, feed_dict={FEAS: ir_feas_output})

					# for c in range(24):
					# 	change_fea = ir_feas_output.copy()
					# 	change_fea[:,:,:,c] = vis_feas_output[:,:,:,c].copy()
					# 	pro_ir_change = sess.run(prob, feed_dict = {FEAS:change_fea})

					'''mean strategy'''
					# print("mean strategy")
					# fuse_feas = 0.5 * vis_feas_output + 0.5 * ir_feas_output

					'''addition strategy'''
					# print("addition strategy")
					# fuse_feas = vis_feas_output + ir_feas_output

					'''max strategy'''
					# print("max strategy")
					# Difference = vis_feas_output - ir_feas_output
					# ders_fuse_vis = np.int8(Difference > 0)
					# ders_fuse_ir = np.int8(Difference < 0)
					# fuse_feas = np.maximum(vis_feas_output, ir_feas_output)

					'''l1_norm strategy'''
					# print("l1_norm strategy")
					# fuse_feas, ders_fuse_vis, ders_fuse_ir = L1_norm(vis_feas_output, ir_feas_output)

					'''CS-based strategy'''
					samples = 30
					ders_vis = np.zeros([samples, H, W, 24])
					ders_ir = np.zeros([samples, H, W, 24])
					diff_vis_ir = ir_img - (vis_img - np.mean(vis_img) + np.mean(ir_img))
					diff_ir_vis = vis_img - np.mean(vis_img) + np.mean(ir_img) - ir_img

					var_vis_pro = np.zeros(shape=(1, samples), dtype=np.float32)
					for i in range(samples):
						var_ir = vis_img - np.mean(vis_img) + np.mean(ir_img) + diff_vis_ir * (i + 1) / samples
						ders_list = sess.run(tf.gradients(out[0, 1], FEAS),
											 feed_dict={FEAS: sess.run(feas, feed_dict={SOURCE: var_ir})})
						ders_ir[i, :, :, :] = np.abs(ders_list[0]) * (0.6 ** i)

						var_vis = ir_img + (i + 1) / samples * diff_ir_vis
						ders_list = sess.run(tf.gradients(out[0, 0], FEAS),
											 feed_dict={FEAS: sess.run(feas, feed_dict={SOURCE: var_vis})})
						ders_vis[i, :, :, :] = np.abs(ders_list[0]) * (0.6 ** i)

						# show saturation regions
						pro_vis = sess.run(prob, feed_dict={FEAS: sess.run(feas, feed_dict={SOURCE: var_vis})})
						if (i + 1) % 5 == 0:
							print("Integral process： step: [%s/%s]" % (i + 1, samples))
						var_vis_pro[0, i] = np.copy(pro_vis[0, 0])

					ders_vis = np.expand_dims(np.mean(ders_vis, axis=0), axis=0)
					ders_ir = np.expand_dims(np.mean(ders_ir, axis=0), axis=0)

					mean_vis = np.mean(ders_vis)
					mean_ir = np.mean(ders_ir)
					ders_ir = ders_ir + mean_vis - mean_ir
					c = 0.00015
					ders_fuse_vis, ders_fuse_ir = softmax(ders_vis / c, ders_ir / c)
					fuse_feas = ders_fuse_vis * vis_feas_output + ders_fuse_ir * ir_feas_output

					result = sess.run(RESULT, feed_dict={FEAS: fuse_feas})
					imsave('results/'+ str(index) + '.jpg', result[0, :, :, 0] / 2 + 0.5)

					end = time.time()
					t.append(end - begin)
				print("mean:%s, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()
