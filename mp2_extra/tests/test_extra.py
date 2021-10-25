import unittest, h5py, extra
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(2)
    def test_extra_SNR_better_than_0_dB(self):
        with h5py.File('extra_solutions.hdf5','r') as f:
            highres_video = extra.animate(f['highres_frames'][:], f['lowres_video'][:])
            SNR = 10*np.log10(np.sum(np.square(f['highres_ref'][:]))/np.sum(np.square(highres_video-f['highres_ref'][:])))
            self.assertGreater(SNR, 0, msg='SNR was not greater than 0dB')

    @weight(2)
    def test_extra_SNR_better_than_10_dB(self):
        with h5py.File('extra_solutions.hdf5','r') as f:
            highres_video = extra.animate(f['highres_frames'][:], f['lowres_video'][:])
            SNR = 10*np.log10(np.sum(np.square(f['highres_ref'][:]))/np.sum(np.square(highres_video-f['highres_ref'][:])))
            self.assertGreater(SNR, 10, msg='SNR was not greater than 10dB')

    @weight(1)
    def test_extra_SNR_better_than_20_dB(self):
        with h5py.File('extra_solutions.hdf5','r') as f:
            highres_video = extra.animate(f['highres_frames'][:], f['lowres_video'][:])
            SNR = 10*np.log10(np.sum(np.square(f['highres_ref'][:]))/np.sum(np.square(highres_video-f['highres_ref'][:])))
            self.assertGreater(SNR, 20, msg='SNR was not greater than 20dB')

