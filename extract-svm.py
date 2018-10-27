import numpy as np

import util
import tqdm
import svm

doi_path = 'data/DOI'
ktrans_path = 'data/ProstateXKtrains-train-fixed'
findings_csv_path = 'data/ProstateX-Findings-Train.csv'
images_csv_path = 'data/ProstateX-Images-Train.csv'

patient_images = util.load_patient_images(findings_csv_path, images_csv_path, doi_path, ktrans_path, \
                                          progress=tqdm.tqdm)

bigX, bigY, bigX_aux = svm.prepare_images(patient_images, flat=False, reduce_mean=False)
bigX_flat, bigY = svm.prepare_images(patient_images, flat=True, reduce_mean=False)

print(bigX.shape)

np.save('/tmp/prostatex-train-5x5mm-1mmpx-1layer.npy', bigX)
np.save('/tmp/prostatex-train-aux.npy', bigX_aux)
np.save('/tmp/prostatex-train.npy', bigX_flat)
np.save('/tmp/prostatex-train-labels.npy', bigY)
