import matplotlib.pyplot as plt

import numpy as np
from numpy import newaxis

import tensorflow as tf

import gan

patch_px = 16
patch_mm = 16
batch_size = 200

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

saver = tf.train.Saver()

saver.restore(sess, 'chk/gan-5000')

bigX_hat = model.bigX_hat.eval(session=sess)

fig, axs = plt.subplots(5, 6, figsize=(6, 5))
axf = axs.flatten()
im = bigX_hat

for ax in axf:
    ax.tick_params(axis='both',
                   top=False, bottom=False, left=False, right=False,
                   labeltop=False, labelbottom=False,
                   labelleft=False, labelright=False)

for c in range(3):
    for i, ax in enumerate(axf):
        ax.imshow(im[i,:,:,c], interpolation='nearest', vmin=0, vmax=1., cmap=plt.cm.plasma)

    fig.savefig('figure-generated-{}.pdf'.format(c))
