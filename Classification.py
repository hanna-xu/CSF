import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.2


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(1, 32, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(64, 128, 3, scope = 'conv1_3'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, Scope = self.scope + '/encoder/b' + str(i), training=is_training)
		return out


class Classification(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('classification'):
				self.weight_vars.append(self._create_variables(128, 32, 7, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(32, 32, 7, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(32, 16, 7, scope = 'conv1_3'))
				self.weight_vars.append(self._create_variables(16, 8, 7, scope = 'conv1_4'))
				self.weight_vars.append(self._create_variables(2, 2, 7, scope = 'conv1_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def classification(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, BN=False, Scope = self.scope + '/classification/b' + str(i), training = is_training)
			out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'+str(i))
			if i == 2:
				out = tf.reduce_mean(out, axis = [1, 2])
		return out



def conv2d(x, kernel, bias, use_lrelu = True, Scope = None, BN = True, training = True, strides = [1, 1, 1, 1]):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = training)
	if use_lrelu:
		out = tf.maximum(out, 0.2*out)
	# out = tf.nn.relu(out)
	return out