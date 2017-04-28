import os

import numpy as np
from numpy import newaxis

import tensorflow as tf

import tqdm

import random
import util
import gan
import run_gan

doi_path = 'data/DOI'
ktrans_path = 'data/ProstateXKtrains-train-fixed'
findings_csv_path = 'data/ProstateX-Findings-Train.csv'
images_csv_path = 'data/ProstateX-Images-Train.csv'

patient_images = util.load_patient_images(findings_csv_path, images_csv_path, doi_path, ktrans_path, progress=tqdm.tqdm)

patch_px = 16
patch_mm = 16
batch_size = 200
n_iters = 5000

model = gan.build_gan(
    gan.build_generator,
    gan.build_discriminator,
    batch_size = batch_size,
    h = patch_px, w = patch_px,
    n_channels = 3,
    n_classes = 2,
    label_smoothing = 0,
    multiclass = True
)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

bigX, bigY, bigX_zone = run_gan.prepare_images_aug(patient_images, patch_px=patch_px, patch_mm=patch_mm)

it = run_gan.batch_generator_one_hot(bigX, bigY, batch_size=batch_size)

saver = tf.train.Saver()

gan.train_gan(n_iters=n_iters,
              batch_iterator=it,
              session=sess,
              model=model,
              progress=tqdm.tqdm,
              g_step_ratio=1,
              plot=False)

save_path = 'chk/gan'

directory = os.path.dirname(save_path)
if not os.path.exists(directory):
    os.makedirs(directory)

saver.save(sess=sess, save_path=save_path, global_step=n_iters)
