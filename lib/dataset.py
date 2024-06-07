import copy
import logging
from typing import Iterator, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Patch
from skimage import transform
import highdicom as hd
import pydicom

from lib.dcmutils import main_orientation_axis

log = logging.getLogger(__name__)


class Radcttace:
    def __init__(self, path, only_patients=None):
        self.path = path
        df = pd.read_excel(path / 'objective_function.ods', engine='odf')
        self.os_d = {f"{int(row['Patient']):04d}": row['OS (D)'] for idx, row in df.iterrows()}
        self.recist = {f"{int(row['Patient']):04d}": row['mRECIST target-lesions post 1ºTACE'] for idx, row in df.iterrows()}
        if only_patients is None:
            self.patients = [Patient(self, p) for p in self.path.iterdir() if p.is_dir()]
        else:
            self.patients = only_patients
        self.patients.sort(key=lambda p: int(p.name))

    def split(self, ratio: float, randomize: bool = False):
        """ Splits the dataset into two datasets. """
        p_dirs = copy.deepcopy(self.patients)
        n = int(ratio * len(p_dirs))
        if randomize:
            np.random.shuffle(p_dirs)
        return Radcttace(self.path, only_patients=p_dirs[:n]), Radcttace(self.path, only_patients=p_dirs[n:])


class Patient:
    def __init__(self, dataset, patient_path):
        self.dataset = dataset
        self.path = patient_path
        self.name = patient_path.name

    def acquisitions(self) -> Iterator['Acquisition']:
        """ Iterates over acquisitions in the patient. """
        acq_dirs = [p for p in self.path.iterdir() if p.is_dir()]
        acq_dirs.sort(key=lambda p: int(p.name[:2]))
        for acq_path in acq_dirs:
            yield Acquisition(self, acq_path)

    def main_acquisition(self) -> 'Acquisition':
        """ Returns the only acquisition which has segmentations. """
        for acq in self.acquisitions():
            try:
                acq.segmentation_tumor()
                return acq
            except ValueError:
                pass

    def objective_function(self):
        """ Returns the objective function of the patient. """
        return self.dataset.recist[self.name]

    def __str__(self):
        return f'{self.name}'

class Acquisition:
    def __init__(self, patient, acq_path):
        self.patient = patient
        self.path = acq_path

    def ct(self) -> 'DicomDir':
        """ Returns the CT image. """
        return DicomDir(self, self.path)

    def segmentation_tumor(self) -> 'Image':
        """ Returns the tumor segmentation. """
        segmentation_path = self.path.parent / f'{self.path.name}_ManualROI_Tumor.dcm'
        if not segmentation_path.exists():
            raise ValueError(f'Segmentation {segmentation_path} does not exist.')
        return DicomSeg(self, path_dicomseg=segmentation_path, ct=self.ct())

    def segmentation_liver(self) -> 'Image':
        """ Returns the liver segmentation. """
        segmentation_path = self.path.parent / f'{self.path.name}_ManualROI_Liver.dcm'
        if not segmentation_path.exists():
            raise ValueError(f'Segmentation {segmentation_path} does not exist.')
        return DicomSeg(self, segmentation_path, ct=self.ct())

    def plot_slice(self, ax: plt.Axes, axis: int = 2, idx_slice: Optional[int] = None,
                   masks: Optional[dict] = None,
                   vmin: Optional[int] = -100, vmax: Optional[int] = 400):
        """ Plots a slice from the CT image.
        * Masks: dict {'name', mask} where mask is a boolean array with the same shape as the CT image. Default: None (liver/tumor segs)
        * Axis: axis to plot the slice according to LFS. Default: 2 (axial)
        """
        ax.set_axis_off()

        img = self.ct().pixel_array()
        if idx_slice is None:
            idx_slice = img.shape[axis] // 2

        log.info(f'{self} > Plotting slice {idx_slice} from axis {axis}.')

        img_slice = img.take(indices=idx_slice, axis=axis)
        px_spacing = self.ct().pixel_spacing_mm()
        px_spacing.pop(axis)
        aspect = px_spacing[0] / px_spacing[1]
        norm = matplotlib.colors.Normalize(vmin=vmin if vmin is not None else np.min(img_slice),
                                           vmax=vmax if vmax is not None else np.max(img_slice))
        img_cmapped = matplotlib.colormaps['bone'](norm(img_slice))[..., :3]

        if masks is None:
            try:
                masks = {
                    'Liver': self.segmentation_liver().pixel_array(),
                    'Tumor': self.segmentation_tumor().pixel_array(),
                }
            except ValueError:
                masks = {}

        if len(masks) == 0:
            ax.imshow(img_cmapped, aspect=aspect)
        else:
            mask_slice = np.zeros(shape=img_slice.shape, dtype=int)
            masks_list = [(name, mask) for name, mask in masks.items()]
            masks_list.sort(key=lambda x: x[1].sum(), reverse=True)

            cmap = plt.colormaps['Set1']
            legend_handles = []
            legend_labels = []
            for idx_mask, (mask_name, mask) in enumerate(masks_list):
                mask_slice[mask.take(indices=idx_slice, axis=axis)] = idx_mask + 1
                color = cmap(idx_mask + 1)
                legend_handles.append(Patch(color=color, label=mask_name))
                legend_labels.append(mask_name)

            mask_nonzero = np.tile((mask_slice > 0)[..., np.newaxis], [1, 1, 3])
            mask_cmapped = cmap(mask_slice)[..., :3]
            mask_cmapped[~mask_nonzero] = 0

            # Legend
            ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')

            return ax.imshow(img_cmapped * (1 - 0.5 * mask_nonzero) + mask_cmapped * 0.5 * mask_nonzero, aspect=aspect)

    def plot_gif(self, fig: plt.Figure, ax: plt.Axes, axis: int = 2, num_slices: Optional[int] = 32,
                   masks: Optional[dict] = None,
                   vmin: Optional[int] = -100, vmax: Optional[int] = 400):
        """ Plots a gif with slices from the CT image.
        * Masks: see `plot_slice`.
        * num_slices: None plots all of them. Default: 32."""
        if masks is None:
            # Compute them only once
            masks = {
                'Liver': self.segmentation_liver().pixel_array(),
                'Tumor': self.segmentation_tumor().pixel_array(),
            }

        if vmin is None:
            vmin = np.min(self.ct().pixel_array())
        if vmax is None:
            vmax = np.max(self.ct().pixel_array())

        sh = self.ct().pixel_array().shape
        if num_slices is None:
            num_slices = sh[axis]
        else:
            num_slices = min(num_slices, sh[axis])

        fig, ax = plt.subplots()
        animation_data = [
            [self.plot_slice(ax, axis=axis, idx_slice=idx, masks=masks, vmin=vmin, vmax=vmax)]
            for idx in np.linspace(0, sh[axis]-1, num_slices).astype(int)
        ]
        anim = animation.ArtistAnimation(fig, animation_data, interval=250, blit=True)
        return anim

    def __str__(self):
        return f'{self.patient} {self.path.name}'

class Image:
    def __init__(self, acquisition):
        self.acquisition = acquisition

    def pixel_array(self):
        """ Returns the pixel array of the segmentation. """
        raise NotImplementedError

    def pixel_spacing_mm(self):
        """ Returns the pixel width for each dimension. """
        raise NotImplementedError

    def pixel_array_resliced(self, new_spacing_mm):
        """ Returns the pixel array resliced to new_spacing_mm. """
        img = self.pixel_array()
        current_spacing_mm = self.pixel_spacing_mm()
        sz = list(img.shape)
        desired_sz = [int(1 + (sz[idx] - 1) * current_spacing_mm()[idx] / new_spacing_mm[idx])
                      for idx in (0, 1, 2)]     # +-1 since we account for spaces between slices, not # of slices.
        # Apply reslicing
        img = transform.resize(img, desired_sz, mode='edge')
        return img


class DicomDir(Image):
    def __init__(self, acquisition, path_dicomdir):
        super().__init__(acquisition)
        file_paths = [p for p in path_dicomdir.iterdir() if p.is_file() and p.suffix == '.dcm']
        self.dcmset = [pydicom.read_file(p) for p in file_paths]
        self.dcmset = [dcm for dcm in self.dcmset if main_orientation_axis(dcm.ImageOrientationPatient)[1] == 'axial']    # Only axial slices
        self.dcmset.sort(key=lambda x: x['ImagePositionPatient'][2].real)                   # Sorted slices.
        assert len(set(dcm.AcquisitionNumber for dcm in self.dcmset)) == 1, 'All slices must belong to the same acquisition.'
        assert len(set(dcm.SliceThickness for dcm in self.dcmset)) == 1, 'All slices must hace the same Slice Thickness.'

    def pixel_array(self):
        """ Returns the pixel array of the segmentation. """
        return np.stack([dcm.pixel_array for dcm in self.dcmset], axis=-1)   # Since slices are axial and sorted.

    def pixel_spacing_mm(self):
        """ Returns the pixel width for each dimension. """
        ps = self.dcmset[0].PixelSpacing
        st = self.dcmset[0].SliceThickness
        return [float(ps[0]), float(ps[1]), float(st)]  # Since it has axial orientation


class DicomSeg(Image):
    def __init__(self, acquisition, path_dicomseg, ct: DicomDir):
        super().__init__(acquisition)
        self.dcmseg = hd.seg.segread(path_dicomseg)
        self.ct = ct
        # Assertions
        orientation = self.dcmseg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
        assert main_orientation_axis(orientation)[1] == 'axial', 'Only axially-sliced segmentations are supported.'

    def pixel_array(self):
        """ Returns the pixel array of the segmentation. """
        dcmseg_pixelarray = np.moveaxis(self.dcmseg.pixel_array, [0, 1, 2], [2, 0, 1])  # Set to LFS orientation
        mask_cropped = dcmseg_pixelarray == self.dcmseg.get_segment_numbers()[0]
        mask = np.zeros(shape=(dcmseg_pixelarray.shape[0], dcmseg_pixelarray.shape[1], len(self.ct.dcmset)), dtype=bool)
        idx_by_positionpatient = {
            ct_dcm.ImagePositionPatient[2]: idx
            for idx, ct_dcm in enumerate(self.ct.dcmset)
        }
        for idx_seg, slice in enumerate(self.dcmseg.PerFrameFunctionalGroupsSequence):
            try:
                idx_ct = idx_by_positionpatient[slice.PlanePositionSequence[0].ImagePositionPatient[2]]
                mask[:, :, idx_ct] = mask_cropped[:, :, idx_seg]
            except KeyError:
                # TODO: interpolate idx as best efforts-approach?
                log.error(f'{self.acquisition} > Slice {idx_seg} from segmentation not found in CT.')
        return mask

    def pixel_spacing_mm(self):
        """ Returns the pixel width for each dimension. """
        ps = self.dcmseg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        st = self.dcmseg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        return [float(ps[0]), float(ps[1]), float(st)]  # Since it has axial orientation
