import unittest

import util
import svm
import gan

class LoadPatientImagesTestCase(unittest.TestCase):
	def setUp(self):
		self.limit = None

	def test_load_training(self):
		doi_path = 'data/DOI'
		ktrans_path = 'data/ProstateXKtrains-train-fixed'
		findings_path = 'data/ProstateX-Findings-Train.csv'

		training_images = util.load_patient_images(
		  findings_path, doi_path, ktrans_path, limit=self.limit)

	def test_load_test(self):
		doi_path = 'data/test/DOI'
		ktrans_path = 'data/test/ProstateXKtrans-test-fixedv2'
		findings_path = 'data/test/ProstateX-Findings-Test.csv'

		test_images = util.load_patient_images(
		  findings_path, doi_path, ktrans_path, limit=self.limit)

class BuildGanTestCase(unittest.TestCase):
	def test_build(self):
		model = gan.build_gan(gan.build_generator, gan.build_discriminator,
		          batch_size=200,
		          h=16, w=16, n_channels=3, n_classes=3)
		for field in model:
			assert field is not None


if __name__ == '__main__':
	unittest.main()
