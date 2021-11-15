# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from networks import Encoder, Classification, Decoder
import scipy.misc
import time
import scipy.io as scio
import scipy.ndimage
import matplotlib.pyplot as plt


def scale(A):
	A = (A - np.min(A)) / (np.max(A) - np.min(A))
	return A

def scale2(A, B):
	b, h, w, c = A.shape
	M = np.max([np.max(A), np.max(B)])
	N = np.min([np.min(A), np.min(B)])
	A = (A - N) / (M - N)
	B = (B - N) / (M - N)
	return A, B


def softmax(x, y):
	x_exp = np.exp(x)
	y_exp = np.exp(y)
	x_out = x_exp / (x_exp + y_exp)
	y_out = y_exp / (x_exp + y_exp)
	return (x_out, y_out)

def count():
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	return total_parameters


def L1_norm(source_en_a, source_en_b):
	narry_a = source_en_a
	narry_b = source_en_b
	dimension = source_en_a.shape

	# caculate L1-norm
	temp_abs_a = tf.abs(narry_a)
	temp_abs_b = tf.abs(narry_b)
	_l1_a = tf.reduce_sum(temp_abs_a, 3)
	_l1_b = tf.reduce_sum(temp_abs_b, 3)

	_l1_a = tf.reduce_sum(_l1_a, 0)
	_l1_b = tf.reduce_sum(_l1_b, 0)
	l1_a = _l1_a.eval()
	l1_b = _l1_b.eval()
	# caculate the map for source images
	mask_value = l1_a + l1_b
	mask_sign_a = l1_a / mask_value
	mask_sign_b = l1_b / mask_value
	array_MASK_a = mask_sign_a
	array_MASK_b = mask_sign_b
	print(array_MASK_a.shape)

	for i in range(dimension[3]):
		temp_matrix = array_MASK_a * narry_a[0, :, :, i] + array_MASK_b * narry_b[0, :, :, i]
		temp_matrix = temp_matrix.reshape([1, dimension[1], dimension[2], 1])
		if i == 0:
			result = temp_matrix
		else:
			result = np.concatenate([result, temp_matrix], axis = -1)
	return result, array_MASK_a.reshape([1, dimension[1], dimension[2], 1]), array_MASK_b.reshape([1, dimension[1], dimension[2], 1])



def save_images(paths, datas, save_path, prefix = None, suffix = None):
	if isinstance(paths, str):
		paths = [paths]

	assert (len(paths) == len(datas))

	if not exists(save_path):
		mkdir(save_path)

	if prefix is None:
		prefix = ''
	if suffix is None:
		suffix = ''

	for i, path in enumerate(paths):
		data = datas[i]
		# print('data ==>>\n', data)
		if data.shape[2] == 1:
			data = data.reshape([data.shape[0], data.shape[1]])
		# print('data reshape==>>\n', data)

		name, ext = splitext(path)
		name = name.split(sep)[-1]

		path = join(save_path, prefix + suffix + ext)
		print('data path==>>', path)
		imsave(path, data)


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g