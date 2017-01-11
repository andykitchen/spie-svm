import os
from glob import glob
from os.path import join
from collections import namedtuple

import numpy as np
import pandas as pd

import dicom
import SimpleITK as sitk

from tqdm import tqdm

def read_series_description(series_path):
	"""for a series directory, return description by reading arbitary DICOM file"""
	f = glob(join(series_path, '*.dcm'))[0]
	d = dicom.read_file(f, stop_before_pixels=True)
	return d.SeriesDescription

def read_patient_descriptions(patient, doi_path):
	"""for a patient, return list of series paths and descriptions"""
	patient_series = glob(join(doi_path, patient, '**/*'))
	return ((series_path, read_series_description(series_path))
				for series_path in patient_series)

def read_dicom_series(series_path):
	"""read all DICOM images in a series directory"""
	reader = sitk.ImageSeriesReader()
	files = reader.GetGDCMSeriesFileNames(series_path)
	reader.SetFileNames(files)
	image = reader.Execute()
	return image

def ktrans_path_for_patient(patient_id, ktrans_path):
	return join(ktrans_path, patient_id, patient_id + '-Ktrans.mhd')

def read_ktrans_image(path):
	"""read in a k-trans image for a given patient"""
	return sitk.ReadImage(path)

def read_image_with_format(path):
	if path.endswith('.mhd'):
		return read_ktrans_image(path)
	elif os.path.isdir(path):
		return read_dicom_series(path)
	else:
		raise

def extract_patch(image, position, patch_pixels=128, patch_mm=60, layers=1, layer_spacing_mm=3., augment=False, sigma=1.0, sigma_theta=0.1):
	"""extract a patch of an image around a position"""

	resampleFilter = sitk.ResampleImageFilter()
	resampleFilter.SetOutputPixelType(sitk.sitkFloat32)
	resampleFilter.SetReferenceImage(image)

	c = np.array(position)
	
	if augment:
		c += sigma*np.random.randn(3)
		tr = sitk.Euler3DTransform()
		a = np.mod(sigma_theta*np.random.randn(3), 2*np.pi)
		# a[2] = 2*np.pi*np.random.rand()
		tr.SetCenter(c)
		tr.SetRotation(*a)
		resampleFilter.SetTransform(tr)

	pxy  = patch_pixels
	pz   = layers
	p    = np.array([pxy, pxy, pz], dtype=np.int)

	resampleFilter.SetSize(p)

	bxy  = patch_mm # mm
	bz   = layer_spacing_mm * (layers // 2)
	b    = np.array([bxy, bxy, bz], dtype=np.float)
	
	s = b / p
	s[2] = layer_spacing_mm

	resampleFilter.SetOutputSpacing(s)

	v = image.GetDirection()
	v = np.array(v).reshape((3, 3))

	cc = c - 0.5*np.dot(v, b)
	
	resampleFilter.SetOutputOrigin(cc)
	
	patch = resampleFilter.Execute(image)
	return patch

def image_to_ndarray_patch(image, pos, **args):
	patch = extract_patch(image, pos, **args)
	nd = sitk.GetArrayFromImage(patch)
	return nd

def images_to_patches(images, pos, **args):
	xs = [image_to_ndarray_patch(image, pos, **args) for image in images]
	return np.concatenate(xs, axis=0)

def lookup_keys(d, keys):
	for k in keys:
		if k in d:
			return d[k]
	raise KeyError

series_desc_lookup = [
	[
		't2_tse_tra',
		't2_tse_tra_Grappa3', # ProstateX-0191
		't2_tse_tra_320_p2' # ProstateX-0218
	],
	[
		'ep2d_diff_tra_DYNDIST_ADC',
		'ep2d_diff_tra_DYNDIST_MIX_ADC',
		'diffusie-3Scan-4bval_fs_ADC',
		'ADC_S3_1', # ProstateX-0191
		'ep2d-advdiff-MDDW-12dir_spair_511b_ADC', # ProstateX-0218
		'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC', # ProstateX-0227
		'diff tra b 50 500 800 WIP511b alle spoelen_ADC' # ProstateX-0240
	],
]

def get_paths_for_patient(patient_id, doi_path, ktrans_path):

	# FIXME sometimes scans have multiple instances with the same
	# description, Jarrel thinks we should the most recent one

	paths = []
	
	path_by_desc = {desc: path 
					for path, desc 
					in read_patient_descriptions(patient_id, doi_path)}

	for keys in series_desc_lookup:
		try:
			series_path = lookup_keys(path_by_desc, keys)
		except KeyError:
			raise KeyError('{}: {}'.format(patient_id, keys))
		paths.append(series_path)

	paths.append(ktrans_path_for_patient(patient_id, ktrans_path))

	return paths

Patient = namedtuple('Patient',
	['id', 'pos', 'clin_sig', 'zone', 'images'])

def load_patient_images(findings_path, doi_path, ktrans_path, progress=tqdm, limit=None):

	# FIXME it seems that using the world matrix and/or IJK
	# in the images CSV leads to different images than using
	# pos directly

	findings_df = pd.read_csv(findings_path)
	findings_df['pos'] = findings_df['pos'].map(lambda x: np.array(map(float, x.split())))

	if limit:
		findings_df = findings_df[:limit]

	def process_row(row):
		patient_id = row['ProxID']
		pos = row['pos']
		if 'ClinSig' in row:
			clin_sig = row['ClinSig']
		else:
			clin_sig = None
		zone = row['zone']

		paths = get_paths_for_patient(patient_id, doi_path, ktrans_path)
		images = [read_image_with_format(path) for path in paths]

		patient_data = Patient(
			id=patient_id,
			pos=pos,
			clin_sig=clin_sig,
			zone=zone,
			images=images)

		return patient_data

	it = progress(findings_df.iterrows(), total=len(findings_df))
	patient_images = [process_row(row) for ix, row in it]

	return patient_images
