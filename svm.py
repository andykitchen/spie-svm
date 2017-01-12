import numpy as np

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.grid_search import GridSearchCV

import util

def prepare_images(patient_images):
	bigXY = [(util.images_to_patches(images, pos, patch_px=5, patch_mm=5, layers=1), clin_sig, zone)
	         for patient_id, pos, clin_sig, zone, images in patient_images]

	bigY = np.array([label if label is not None else -1 for data, label, zone in bigXY], dtype=np.int)
	bigX_zone = [zone for data, label, zone in bigXY]
	zone_to_index = {zone: i for i, zone in enumerate(sorted(list(set(bigX_zone))))}
	bigX_zone = np.array([zone_to_index[zone] for zone in bigX_zone], dtype=np.int)

	bigX = np.concatenate([pixels.transpose(1,2,0)[np.newaxis,:] for pixels, label, zone in bigXY], axis=0)

	bigX_z = bigX.copy()
	bigX_z[:,:,:,2] = np.log(bigX[:,:,:,2] + 1e-2)

	bigX_z -= bigX_z.mean(axis=0, keepdims=True)
	bigX_z /= bigX_z.std(axis=0, keepdims=True)

	bigX_flat = bigX_z.reshape(bigX_z.shape[0], -1)

	from sklearn.preprocessing import OneHotEncoder

	enc = OneHotEncoder()
	bigX_zone_hot = enc.fit_transform(bigX_zone[:, np.newaxis])
	bigX_zone_hot = bigX_zone_hot.toarray()

	bigX_flat = np.hstack((bigX_flat, bigX_zone_hot))

	return bigX_flat, bigY

def train_svm(patient_images):
	bigX_flat, bigY = prepare_images(patient_images)


	model = svm.SVC(class_weight='balanced')

	auc_scorer = make_scorer(roc_auc_score, needs_threshold=True)

	# model.fit(bigX_flat, bigY)
	# scores = cross_val_score(model, bigX_flat, bigY, cv=3, scoring=auc_scorer)
	# print scores

	params = {
	    'C': [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50],
	    'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001, 0.00001],
	}

	gcv = GridSearchCV(model, param_grid=params, scoring=auc_scorer, cv=3)
	gcv.fit(bigX_flat, bigY)

	return gcv

if __name__ == "__main__":
	doi_path = 'data/DOI'
	ktrans_path = 'data/ProstateXKtrains-train-fixed'
	findings_path = 'data/ProstateX-Findings-Train.csv'

	patient_images = util.load_patient_images(findings_path, doi_path, ktrans_path)
	gcv = train_svm(patient_images)

	print(gcv.best_score_)
	print(gcv.best_params_)

