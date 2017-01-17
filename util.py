import os
import random
from glob import glob
from os.path import join
from collections import namedtuple

import numpy as np
from numpy import newaxis
import pandas as pd

import dicom
import SimpleITK as sitk

from sklearn.feature_extraction.image import extract_patches_2d

from tqdm import tqdm

def read_dicom_series(series_path, world_matrix):
	"""read all DICOM images in a series directory"""
	reader = sitk.ImageSeriesReader()
	files = reader.GetGDCMSeriesFileNames(series_path)
	reader.SetFileNames(files)
	image = reader.Execute()

	# NB
	s = np.identity(4)
	s[:3, :3] = np.diag(1./np.array(image.GetSpacing()))
	sm = np.dot(world_matrix, s)

	d = image.GetDirection()
	d = np.matrix(d).reshape(3, 3)
	assert np.allclose(d, sm[:3, :3])

	# NB
	image.SetOrigin(world_matrix[:3, -1].flatten().tolist()[0])

	return image

def ktrans_path_for_patient(patient_id, ktrans_path):
	return join(ktrans_path, patient_id, patient_id + '-Ktrans.mhd')

def read_ktrans_image(path):
	"""read in a k-trans image for a given patient"""
	return sitk.ReadImage(path)

def read_image_with_format(path, world_matrix):
	if path.endswith('.mhd'):
		return read_ktrans_image(path)
	elif os.path.isdir(path):
		return read_dicom_series(path, world_matrix)
	else:
		raise

def extract_patch(
	image, position,
	patch_px=128, patch_mm=60,
	layers=1, layer_spacing_mm=3.,
	snap_to_grid=False,
	augment=False, sigma=1.0, sigma_theta=0.1):
	"""extract a patch of an image around a position"""

	resampleFilter = sitk.ResampleImageFilter()
	resampleFilter.SetOutputPixelType(sitk.sitkFloat32)
	resampleFilter.SetReferenceImage(image)

	c = np.array(position)

	if snap_to_grid:
		ix = image.TransformPhysicalPointToIndex(c)
		c = image.TransformIndexToPhysicalPoint(ix)

	if augment:
		c += sigma*np.random.randn(3)
		tr = sitk.Euler3DTransform()
		a = np.mod(sigma_theta*np.random.randn(3), 2*np.pi)
		# a[2] = 2*np.pi*np.random.rand()
		tr.SetCenter(c)
		tr.SetRotation(*a)
		resampleFilter.SetTransform(tr)

	pxy  = patch_px
	pz   = layers
	p    = np.array([pxy, pxy, pz], dtype=np.int)

	resampleFilter.SetSize(p.tolist())

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

Patient = namedtuple('Patient',
	['id', 'pos', 'clin_sig', 'zone', 'age', 'weight', 'images'])

def load_patient_images(
	findings_csv_path, images_csv_path,
	doi_path, ktrans_path, 
	progress=tqdm, limit=None):

	# FIXME it seems that using the world matrix and/or IJK
	# in the images CSV leads to different images than using
	# pos directly

	findings_df = pd.read_csv(findings_csv_path)
	findings_df['pos'] = findings_df['pos'].map(lambda x: np.array(list(map(float, x.split()))))

	images_df = pd.read_csv(images_csv_path)
	images_df = images_df.set_index(['ProxID', 'DCMSerDescr', 'DCMSerNum'])
	images_df = images_df[~images_df.index.duplicated(keep='first')]
	images_df = images_df.sort_index()

	patients = sorted(findings_df['ProxID'].unique())

	patient_series_metadata = load_patient_series_metadata(
		patients, doi_path, progress)

	patient_ages = {
		k: np.mean([x.age for x in v])
		for k, v in patient_series_metadata.items()}

	patient_weights = {
		k: np.mean([x.weight for x in v])
		for k, v in patient_series_metadata.items()}

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

		paths = get_paths_for_patient(
			patient_id, ktrans_path, patient_series_metadata, images_df)

		images = [read_image_with_format(path, world_matrix)
			for path, world_matrix in paths]

		patient_data = Patient(
			id=patient_id,
			pos=pos,
			clin_sig=clin_sig,
			zone=zone,
			age=patient_ages[patient_id],
			weight=patient_weights[patient_id],
			images=images)

		return patient_data

	it = progress(findings_df.iterrows(), total=len(findings_df))
	patient_images = [process_row(row) for ix, row in it]

	return patient_images

def extract_random_patches(x, max_patches):
	image = x.images[1]
	nd = sitk.GetArrayFromImage(image)
	layer = random.randrange(0, nd.shape[0])
	patches = extract_patches_2d(nd[layer], (16, 16), max_patches=max_patches)
	patches = patches[..., newaxis]
	patches = patches.astype(np.float32) / 3000.
	return patches
	
def random_element_generator(col):
	while True:
		yield random.choice(col)

def concatenate_generator(it, n_elems):
	while True:
		yield np.concatenate([next(it) for i in range(n_elems)])
		
def patch_generator(patient_images, n_images=4, max_patches=50):
	return concatenate_generator(
		map(lambda im: extract_random_patches(im, max_patches=max_patches),
			random_element_generator(patient_images)), n_elems=n_images)

SeriesMetadata = namedtuple('SeriesMetadata', ['desc', 'num', 'path', 'date', 'age', 'weight'])

def get_patient_series(patient, doi_path):
	return glob(join(doi_path, patient, '**/*'))

def assert_all_same(ls):
	assert len(set(ls)) == 1

def parse_age(s):
	assert s[-1] == 'Y'
	return int(s[:-1])

def get_series_dcm_paths(series_path):
	return sorted(glob(join(series_path, '*.dcm')))

def load_windows(series_path):
	assert_all_same(d.SeriesDescription for d in ds)
	assert_all_same(d.SeriesNumber for d in ds)
	assert_all_same(d.PatientWeight for d in ds)
	assert_all_same(d.AcquisitionDate for d in ds)
	ds = [dicom.read_file(f, stop_before_pixels=True) for f in fs]
	windows = [(d.WindowCenter, d.WindowWidth) for d in ds]

def load_series_metadata(series_path):
	f = get_series_dcm_paths(series_path)[0]
	d = dicom.read_file(f, stop_before_pixels=True)
	return SeriesMetadata(
		desc=d.SeriesDescription,
		num=int(d.SeriesNumber),
		path=series_path,
		date=d.AcquisitionDate,
		age=parse_age(d.PatientAge),
		weight=int(d.PatientWeight))

def lookup_series_by_desc(m, patient_id, desc):
	# NB some test dicoms have no image entries
	if patient_id == 'ProstateX-0206':
		m = filter(lambda s: desc in s.desc and s.num != 55, m)
	elif patient_id == 'ProstateX-0224':
		m = filter(lambda s: desc in s.desc and s.num != 6, m)
	elif patient_id == 'ProstateX-0261':
		m = filter(lambda s: desc in s.desc and s.num != 6, m)
	else:
		m = filter(lambda s: desc in s.desc, m)

	m = sorted(m, key=lambda x: (x.date, x.num))[-1]
	return m

def ktrans_path_for_patient(patient_id, ktrans_path):
	return join(ktrans_path, patient_id, patient_id + '-Ktrans.mhd')

def parse_world_matrix(s):
	m = np.fromstring(s, dtype=np.float, sep=',')
	m = m.reshape(4, 4)
	return np.matrix(m)

def lookup_world_matrix(image_details, p_id, desc, num):
	s = image_details.loc[p_id, desc, num].WorldMatrix
	return parse_world_matrix(s)

def load_patient_series_metadata(patients, doi_path, progress=None):
	if progress:
		patients = progress(patients)

	patient_series_metadata = {
		p: [load_series_metadata(s)
			for s in get_patient_series(p, doi_path)]
		for p in patients}

	return patient_series_metadata


def get_paths_for_patient(patient_id, ktrans_path, patient_series_metadata, image_details):
	md      = patient_series_metadata[patient_id]
	t2      = lookup_series_by_desc(md, patient_id, 't2_tse_tra')
	adc     = lookup_series_by_desc(md, patient_id, 'ADC')

	t2_mat  = lookup_world_matrix(image_details, patient_id, t2.desc, t2.num)
	adc_mat = lookup_world_matrix(image_details, patient_id, adc.desc, adc.num)
	ktrans  = ktrans_path_for_patient(patient_id, ktrans_path)

	return [(t2.path, t2_mat), (adc.path, adc_mat), (ktrans, None)]
