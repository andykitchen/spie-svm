import numpy as np
from numpy import newaxis

import pandas as pd

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.grid_search import GridSearchCV

import util

def prepare_images(patient_images):
	bigXY = [(
	  util.images_to_patches(p.images, p.pos, patch_px=1, patch_mm=1, layers=1),
	  p.clin_sig)
	    for p in patient_images]

	bigY = np.array([label if label is not None else -1 for data, label in bigXY], dtype=np.int)
	bigX_zone = [p.zone for p in patient_images]
	zone_to_index = {zone: i for i, zone in enumerate(sorted(list(set(bigX_zone))))}
	bigX_zone = np.array([zone_to_index[zone] for zone in bigX_zone], dtype=np.int)

	bigX_age = np.array([p.age for p in patient_images])
	bigX_age -= bigX_age.mean()
	bigX_age /= bigX_age.std()
	bigX_age = bigX_age[:, newaxis]

	bigX = np.concatenate([pixels.transpose(1,2,0)[newaxis,:] for pixels, label in bigXY], axis=0)

	bigX_z = bigX.copy()

	bigX_z[...,2] = np.log(bigX[...,2] + 1e-2)

	bigX_z -= bigX_z.mean(axis=0, keepdims=True)
	bigX_z /= bigX_z.std(axis=0, keepdims=True)

	bigX_flat = bigX_z.reshape(bigX_z.shape[0], -1)

	from sklearn.preprocessing import OneHotEncoder

	enc = OneHotEncoder()
	bigX_zone_hot = enc.fit_transform(bigX_zone[:, newaxis])
	bigX_zone_hot = bigX_zone_hot.toarray()

	bigX_flat = np.hstack((bigX_flat, bigX_zone_hot, bigX_age))

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
	  'gamma': [1.0, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001],
	  # 'gamma': ['auto', 1., 0.1, 0.01, 0.001, 0.0001, 0.00001],
	}

	gcv = GridSearchCV(model, param_grid=params, scoring=auc_scorer, cv=6)
	gcv.fit(bigX_flat, bigY)

	return gcv

def load_train_data():
	doi_path = 'data/DOI'
	ktrans_path = 'data/ProstateXKtrains-train-fixed'
	findings_csv_path = 'data/ProstateX-Findings-Train.csv'
	images_csv_path = 'data/ProstateX-Images-Train.csv'

	return util.load_patient_images(
	  findings_csv_path, images_csv_path,
	  doi_path, ktrans_path)

def load_test_data():
	doi_path = 'data/test/DOI'
	ktrans_path = 'data/test/ProstateXKtrans-test-fixedv2'
	findings_csv_path = 'data/test/ProstateX-Findings-Test.csv'
	images_csv_path = 'data/test/ProstateX-Images-Test.csv'

	patient_images = util.load_patient_images(
	  findings_csv_path, images_csv_path,
	  doi_path, ktrans_path)

	test_findings_df = pd.read_csv(findings_csv_path)

	return patient_images, test_findings_df

if __name__ == '__main__':

	patient_images = load_train_data()

	model = train_svm(patient_images)

	print(model.best_score_)
	print(model.best_params_)

	test_images, test_findings_df = load_test_data()

	bigX_test, _ = prepare_images(test_images)
	# y_pred = model.predict_proba(bigX_test)
	y_pred = model.decision_function(bigX_test)

	pred_df = pd.DataFrame({
	    'ProxID': test_findings_df['ProxID'],
	    'fid': test_findings_df['fid'],
	    'pred_higher_is_significant': y_pred
	})

	pred_df.to_csv('data/ProstateX-Submission.csv', index=False)
