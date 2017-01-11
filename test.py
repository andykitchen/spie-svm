import unittest

import util
import svm

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

if __name__ == '__main__':
    unittest.main()
