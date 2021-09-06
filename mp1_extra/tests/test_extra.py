import unittest, h5py, extra
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(1)
    def test_extra_better_than_15_percent(self):
        with h5py.File('extra_test.hdf5','r') as f:
            hypvuv = extra.better_vad(f['signal'], 8000)
            errorrate = np.count_nonzero(hypvuv != f['refvuv'])/len(f['refvuv'])
            self.assertLess(errorrate, 0.15)

    @weight(1)
    def test_extra_better_than_12_percent(self):
        with h5py.File('extra_test.hdf5','r') as f:
            hypvuv = extra.better_vad(f['signal'], 8000)
            errorrate = np.count_nonzero(hypvuv != f['refvuv'])/len(f['refvuv'])
            self.assertLess(errorrate, 0.12)

    @weight(1)
    def test_extra_better_than_9_percent(self):
        with h5py.File('extra_test.hdf5','r') as f:
            hypvuv = extra.better_vad(f['signal'], 8000)
            errorrate = np.count_nonzero(hypvuv != f['refvuv'])/len(f['refvuv'])
            self.assertLess(errorrate, 0.09)

    @weight(2)
    def test_extra_better_than_6_percent(self):
        with h5py.File('extra_test.hdf5','r') as f:
            hypvuv = extra.better_vad(f['signal'], 8000)
            errorrate = np.count_nonzero(hypvuv != f['refvuv'])/len(f['refvuv'])
            self.assertLess(errorrate, 0.06)

