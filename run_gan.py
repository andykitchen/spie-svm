import numpy as np
from numpy import newaxis

import util
import gan


def prepare_images(patient_images, **args):
	bigXY = [(util.images_to_patches(p.images, p.pos, **args), p.clin_sig, p.zone)
	          for p in patient_images]

	bigY = np.array([label if label is not None else -1 for data, label, zone in bigXY], dtype=np.int)
	bigX_zone = [zone for data, label, zone in bigXY]
	zone_to_index = {zone: i for i, zone in enumerate(sorted(list(set(bigX_zone))))}
	bigX_zone = np.array([zone_to_index[zone] for zone in bigX_zone], dtype=np.int)

	bigX = np.concatenate([pixels.transpose(1,2,0)[np.newaxis,:] for pixels, label, zone in bigXY], axis=0)

	bigX[...,0] /= 500.
	bigX[...,1] /= 3000.
	bigX[...,2] /= 5.

	return bigX, bigY, bigX_zone


def prepare_images_aug(patient_images, n_aug=5, **args):
	bigX, bigY, bigX_zone = prepare_images(patient_images, **args)

	aug = [prepare_images(patient_images, augment=True, **args)
		 for i in range(n_aug)]

	bigX_aug = np.concatenate([bigX] + [x for x, y, z in aug])
	bigY_aug = np.concatenate([bigY] + [y for x, y, z in aug])


	return bigX_aug, bigY_aug, bigX_zone


def batch_generator(bigX, bigY, batch_size):
	while True:
		ix = np.random.choice(bigX.shape[0], batch_size)
		bigX_batch = bigX[ix]
		bigY_batch = bigY[ix]
		yield bigX_batch, bigY_batch


def batch_generator_one_hot(bigX, bigY, batch_size):
	while True:
		ix = np.random.choice(bigX.shape[0], batch_size)
		bigX_batch = bigX[ix]
		bigY_batch = bigY[ix]
		bigY_batch_one_hot = np.identity(2)[bigY_batch, :]
		yield bigX_batch, bigY_batch_one_hot


def merged_generator(lesion_iter, background_iter):
	while True:
		lesion_images, lesion_labels = next(lesion_iter)
		background_images = next(background_iter)
		lesion_labels_one_hot = np.identity(3)[lesion_labels, :]
		background_labels = np.repeat([[0, 0, 1]], len(background_images), axis=0)
		bigY_merged = np.concatenate((lesion_labels_one_hot, background_labels))
		bigX_merged = np.concatenate((lesion_images, background_images))
		yield (bigX_merged, bigY_merged)
