import math

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
    tumor_interior_mask = erosion_on_bounding_box(liver_dicomseg.pixel_array() & ~tumor_dicomseg.pixel_array(), se)
    return tumor_interior_mask


def tumor_interior(tumor_dicomseg, width_to_erode_mm):
    """ Tumor without the border. """
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_to_erode_mm)
    tumor_interior_mask = erosion_on_bounding_box(tumor_dicomseg.pixel_array(), se)
    return tumor_interior_mask


def liver_core(liver_dicomseg, tumor_dicomseg, radius_mm):
    """ Small region of (healthy?) liver, far from its edges and tumor. """
    liver_mask = liver_dicomseg.pixel_array()
    tumor_mask = tumor_dicomseg.pixel_array()
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), radius_mm)
    center_idx = np.argmax(distance_transform_cdt(liver_mask & ~tumor_mask))
    liver_core_mask = np.zeros_like(liver_mask)
    liver_core_mask[np.unravel_index(center_idx, liver_core_mask.shape)] = 1
    liver_core_mask = dilation_on_bounding_box(liver_core_mask, se)
    return liver_core_mask & liver_mask


def tumor_core(tumor_dicomseg, radius_mm):
    """ Ball centered around center-most part of the tumor. """
    tumor_mask = tumor_dicomseg.pixel_array()
    center_idx = np.argmax(distance_transform_cdt(tumor_mask))
    # Paint on new mask
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), radius_mm)
    tumor_core_mask = np.zeros_like(tumor_mask)
    tumor_core_mask[np.unravel_index(center_idx, tumor_core_mask.shape)] = 1
    tumor_core_mask = dilation_on_bounding_box(tumor_core_mask, se)
    return tumor_core_mask & tumor_mask


def tumor_inner_perimeter(tumor_dicomseg, width_mm):
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_mm)
    tumor_mask = tumor_dicomseg.pixel_array()
    tumor_eroded_mask = erosion_on_bounding_box(tumor_mask, se)
    inner_perimeter_mask = tumor_mask & ~tumor_eroded_mask
    return inner_perimeter_mask


def tumor_outer_perimeter(liver_dicomseg, tumor_dicomseg, width_mm):
    se = SphericalStructuringElement.get(tumor_dicomseg.pixel_spacing_mm(), width_mm)
    tumor_mask = tumor_dicomseg.pixel_array()
    tumor_dilated = dilation_on_bounding_box(tumor_mask, se)
    outer_perimeter_mask = tumor_dilated & ~tumor_mask & liver_dicomseg.pixel_array()
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
        sz = [math.floor(radius_mm / p) for p in pixel_spacing_mm]
        assert all([r >= 1 for r in sz])
        X, Y, Z = np.meshgrid(*[np.arange(-n, n + 1)*mm for n, mm in zip(sz, pixel_spacing_mm)])
        R = np.sqrt(X**2 + Y**2 + Z**2)
        return R <= radius_mm


def erosion_on_bounding_box(mask, se):
    """ Erosion on the bounding box of the mask. """
    bbox = np.argwhere(mask)
    min_idx = np.min(bbox, axis=0)
    max_idx = np.max(bbox, axis=0)
    mask_bbox = mask[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
    eroded_bbox = morphology.erosion(mask_bbox, se)
    mask[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = eroded_bbox
    return mask


def dilation_on_bounding_box(mask, se):
    """ Dilation on the bounding box of the mask. """
    pad = [n // 2 for n in se.shape]
    bbox = np.argwhere(mask)
    min_idx = np.min(bbox, axis=0)
    max_idx = np.max(bbox, axis=0)
    mask_bbox = mask[min_idx[0]-pad[0]:max_idx[0]+pad[0], min_idx[1]-pad[1]:max_idx[1]+pad[1], min_idx[2]-pad[2]:max_idx[2]+pad[2]]
    dilated_bbox = morphology.dilation(mask_bbox, se)
    mask[min_idx[0]-pad[0]:max_idx[0]+pad[0], min_idx[1]-pad[1]:max_idx[1]+pad[1], min_idx[2]-pad[2]:max_idx[2]+pad[2]] = dilated_bbox
    return mask
