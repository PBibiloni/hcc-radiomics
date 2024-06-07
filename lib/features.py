import SimpleITK as sitk
import numpy as np

from radiomics.featureextractor import RadiomicsFeatureExtractor

from lib.dataset import Image


def dummy_features(ct: Image, mask: np.ndarray):
    """ Compute volume and return it."""
    return {
        'volume_mm3': mask.sum() * np.prod(ct.pixel_spacing_mm()),
        'mean_value': ct.pixel_array()[mask].mean(),
    }


def radiomic_features(ct: Image, mask: np.ndarray):
    """ Extracts radiomic features from the CT image using the mask. """
    mask = mask.astype(int)
    mask[~(mask == 0)] = 1
    ct_itk = sitk.GetImageFromArray(ct.pixel_array())
    seg_itk = sitk.GetImageFromArray(mask.astype(int))

    extractor = RadiomicsFeatureExtractor()
    features = extractor.execute(ct_itk, seg_itk)
    return features