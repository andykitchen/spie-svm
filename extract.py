import numpy as np

import util
import tqdm

doi_path = 'data/DOI'
ktrans_path = 'data/ProstateXKtrains-train-fixed'
findings_csv_path = 'data/ProstateX-Findings-Train.csv'
images_csv_path = 'data/ProstateX-Images-Train.csv'

patient_images = util.load_patient_images(findings_csv_path, images_csv_path, doi_path, ktrans_path, \
                                          progress=tqdm.tqdm)

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

patch_px = 16
patch_mm = 16
n_layers = 5
n_chan = 3
layer_spacing = 3.

bigX, bigY, bigX_zone = prepare_images(patient_images, patch_px=patch_px, patch_mm=patch_mm, layers=n_layers, layer_spacing_mm=layer_spacing)
bigX = bigX.reshape([bigX.shape[0], bigX.shape[1], bigX.shape[2], n_chan, n_layers])
bigX = bigX.transpose(0, 1, 2, 4, 3)

print(bigX.shape)

np.save('/tmp/prostatex-16x16mm-1mmpx-5layer.npy', bigX)
np.save('/tmp/prostatex-labels.npy', bigY)