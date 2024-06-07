import numpy as np
from scipy.ndimage import distance_transform_cdt
from skimage import morphology


def whole_liver(liver_dicomseg, tumor_dicomseg):
    return liver_dicomseg.pixel_array() & ~tumor_dicomseg.pixel_array()


def whole_tumor(tumor_dicomseg):
    return tumor_dicomseg.pixel_array()


def liver_interior(liver_dicomseg, tumor_dicomseg, width_to_erode_mm):
    """ Liver without the border with tumor and other structures. """
    se = SphericalStructuringElement.get(liver_dicomseg.pixel_spacing_mm(), width_to_erode_mm)
    tumor_interior_mask = morphology.erosion(liver_dicomseg.pixel_array() & ~tumor_dicomseg.pixel_array(), se)
    return tumor_interior_mask


def tumor_interior(tumor_dicomseg, width_to_erode_mm):
    """ Tumor without the border. """
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_to_erode_mm)
    tumor_interior_mask = morphology.erosion(tumor_dicomseg.pixel_array(), se)
    return tumor_interior_mask


def liver_core(liver_dicomseg, tumor_dicomseg, radius_mm):
    """ Small region of (healthy?) liver, far from its edges and tumor. """
    liver_mask = liver_dicomseg.pixel_array()
    tumor_mask = tumor_dicomseg.pixel_array()
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), radius_mm)
    center_idx = np.argmax(distance_transform_cdt(liver_mask & ~tumor_mask))
    liver_core_mask = np.zeros_like(liver_mask)
    liver_core_mask[center_idx] = 1
    liver_core_mask = morphology.dilation(tumor_dicomseg.pixel_array(), se)
    return liver_core_mask


def tumor_core(tumor_dicomseg, radius_mm):
    """ Ball centered around center-most part of the tumor. """
    tumor_mask = tumor_dicomseg.pixel_array()
    center_idx = np.argmax(distance_transform_cdt(tumor_mask))
    # Paint on new mask
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), radius_mm)
    tumor_core_mask = np.zeros_like(tumor_mask)
    tumor_core_mask[center_idx] = 1
    tumor_core_mask = morphology.dilation(tumor_dicomseg.pixel_array(), se)
    return tumor_core_mask


def tumor_inner_perimeter(tumor_dicomseg, width_mm):
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_mm)
    tumor_core_mask = morphology.erosion(tumor_dicomseg.pixel_array(), se)
    inner_perimeter_mask = tumor_dicomseg.pixel_array() & ~tumor_core_mask
    return inner_perimeter_mask


def tumor_outer_perimeter(liver_dicomseg, tumor_dicomseg, width_mm):
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_mm)
    tumor_dilated = morphology.dilation(tumor_dicomseg.pixel_array(), se)
    outer_perimeter_mask = tumor_dilated & ~tumor_dicomseg.pixel_array() & liver_dicomseg.pixel_array()
    return outer_perimeter_mask


class SphericalStructuringElement:
    """ Encapsulated in class to avoid recomputing the same structuring element. """
    cached = {}

    @classmethod
    def get(cls, pixel_spacing_mm, radius_mm):
        key = (tuple(pixel_spacing_mm), radius_mm)
        if key not in cls.cached:
            cls.cached[key] = cls._create(pixel_spacing_mm, radius_mm)
        return cls.cached[key]

    @staticmethod
    def _create(pixel_spacing_mm, radius_mm):
        """ Returns an anisotropic spherical structuring element with the given radius. """
        radius_px = [radius_mm / p for p in pixel_spacing_mm]
        assert all([r >= 1 for r in radius_px])
        n = int(max(radius_px) + 0.5)
        X, Y, Z = np.meshgrid(*[np.arange(-n, n + 1)*mm for mm in pixel_spacing_mm])
        R = np.sqrt(X**2 + Y**2 + Z**2)
        return R <= max(radius_px)
