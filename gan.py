from collections import namedtuple

import numpy as np
from numpy import newaxis

import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt


alpha = 0.1

def leaky_relu(x):
	return tf.maximum(alpha*x,x)


def gaussian_noise(x, stddev=0.5):
	return tf.add(x, tf.random_normal(shape=x.get_shape(), stddev=stddev))


def build_discriminator(bigX, n_classes=1, is_training=True, sigma=.5):
	net = bigX

	with slim.arg_scope([slim.conv2d, slim.fully_connected],
		   activation_fn=leaky_relu,
		   biases_initializer=tf.constant_initializer(0.0),
		   weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.5)):

		net = slim.conv2d(net, 32, [3,3], scope='conv1')
		net = gaussian_noise(net, sigma)
		net = slim.conv2d(net, 64, [3,3], scope='conv2', stride=[2,2])
		net = gaussian_noise(net, sigma)
		net = slim.conv2d(net, 128, [3,3], scope='conv3', stride=[2,2])
		net = tf.reduce_mean(net, reduction_indices=(1,2), keep_dims=True)
		net = slim.flatten(net, scope='flat')
		net = slim.dropout(net, 0.5, is_training=is_training)
		net = slim.fully_connected(net, n_classes, scope='fc1',
		  activation_fn=None,
		  biases_initializer=tf.constant_initializer(0),
		  weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

	logits = net
	return logits


def build_generator(bigZ, n_channels=1):
	net = bigZ

	with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
		   activation_fn=tf.nn.relu,
		   weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.43),
		   biases_initializer=tf.constant_initializer(0.1)):

		net = slim.fully_connected(net, 4 * 4 * 16, scope='fc1')
		net = tf.reshape(net, [-1, 4, 4, 16])
		net = slim.conv2d_transpose(net, 32, [3,3], stride=[2,2], scope='tconv1')
		net = slim.conv2d_transpose(net, 16, [3,3], stride=[2,2], scope='tconv2')
		net = slim.conv2d_transpose(net, n_channels, [3,3], scope='tconv3',
		  activation_fn=tf.nn.relu,
		  biases_initializer=tf.constant_initializer(0),
		  weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
		
	bigX_hat = net
	return bigX_hat

GanModel = namedtuple('GanModel', [
	  'bigX', 'bigX_hat', 'bigY',
	  'dloss', 'gloss',
	  'train_step_d', 'train_step_g',
	  'logits_real', 'logits_fake',
	  'is_training',
	  'probs_real',
	])


def build_gan(build_generator, build_discriminator,
	batch_size, h, w, n_channels, n_classes=1, label_smoothing=0, multiclass=False):

	n_latent = 25

	bigX = tf.placeholder(tf.float32, shape=[batch_size, h, w, n_channels])
	bigY = tf.placeholder(tf.float32, shape=[batch_size, n_classes])
	bigZ = tf.random_normal(shape=[batch_size, n_latent])

	is_training = tf.placeholder_with_default(True, shape=())

	with tf.variable_scope('G'):
		bigX_hat = build_generator(bigZ, n_channels)

	with tf.variable_scope('D'):
		logits_real = build_discriminator(bigX, n_classes, is_training)

	with tf.variable_scope('D', reuse=True):
		logits_fake = build_discriminator(bigX_hat, n_classes, is_training)

	if multiclass:
		label_zeros = tf.zeros([batch_size, n_classes])
		label_ones  = tf.ones([batch_size, 1])
		logit_zeros = tf.zeros([batch_size, 1])

		logits_real_extended = tf.concat(1, [logits_real, logit_zeros])
		labels_real_extended = tf.concat(1, [bigY, label_ones])

		logits_fake_extended = tf.concat(1, [logits_fake, logit_zeros])
		labels_fake = tf.concat(1, [label_zeros, label_ones])

		dloss_real_batch = slim.losses.softmax_cross_entropy(logits_real_extended, labels_real_extended,
		                                                     label_smoothing=label_smoothing)

		dloss_fake_batch = slim.losses.softmax_cross_entropy(logits_fake_extended, labels_fake,
		                                                     label_smoothing=label_smoothing)

		probs_real = tf.nn.softmax(logits_real_extended)

	else:
		dloss_real_batch = slim.losses.sigmoid_cross_entropy(
		        logits_real, tf.ones([batch_size, 1]),
		        label_smoothing=label_smoothing)

		dloss_fake_batch = slim.losses.sigmoid_cross_entropy(
		        logits_fake, tf.zeros([batch_size, 1]),
		        label_smoothing=label_smoothing)

		probs_real = tf.nn.softmax(logits_real)

	dloss_real = tf.reduce_mean(dloss_real_batch)
	dloss_fake = tf.reduce_mean(dloss_fake_batch)

	dloss = 0.5*(dloss_real + dloss_fake)
	gloss = -dloss_fake

	dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
	gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

	opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.1, epsilon=1e-6)

	train_step_d = opt.minimize(dloss, var_list=dvars)
	train_step_g = opt.minimize(gloss, var_list=gvars)

	model = GanModel(
		bigX = bigX,
		bigX_hat = bigX_hat,
		bigY = bigY,
		dloss = dloss,
		gloss = gloss,
		train_step_d = train_step_d,
		train_step_g = train_step_g,
		logits_real = logits_real,
		logits_fake = logits_fake,
		is_training = is_training,
		probs_real = probs_real
	)

	return model


def train_gan(n_iters, batch_iterator, model,
	session=None, g_step_ratio=1,
	plot=True, plot_every=100, plot_channel=0,
	progress=None):

	from IPython import display

	sess = session if session else tf.get_default_session()

	gloss_val = np.inf
	dloss_val = np.inf

	fig, axs = plt.subplots(5, 6, figsize=(6, 5))
	plt.close(fig)
	axf = axs.flatten()

	for ax in axf:
		ax.tick_params(axis='both',
		               top=False, bottom=False, left=False, right=False,
		               labeltop=False, labelbottom=False,
		               labelleft=False, labelright=False)


	t = progress(range(n_iters)) if progress else range(n_iters)

	for i in t:
		gloss_val, _ = sess.run([model.gloss, model.train_step_g])

		if i % g_step_ratio == 0:
			bigX_batch, bigY_batch = next(batch_iterator)
			dloss_val, gloss_val, _ = sess.run([model.dloss, model.gloss, model.train_step_d],
				feed_dict={
					model.bigX: bigX_batch,
					model.bigY: bigY_batch
				})

		t.set_description('%.2f, %.2f' % (dloss_val, gloss_val))

		if plot and i % plot_every == 0:
			im = model.bigX_hat.eval()

			for i, ax in enumerate(axf):
				# ax.imshow(im[i], interpolation='nearest', vmin=0, vmax=1.)
				ax.imshow(im[i,:,:,plot_channel], interpolation='nearest', vmin=0, vmax=1., cmap=plt.cm.plasma)

			display.clear_output(wait=True)
			display.display(fig)

train_gan_nb = train_gan
