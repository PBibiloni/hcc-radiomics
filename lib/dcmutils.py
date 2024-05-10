import numpy as np
from pydicom.fileset import FileSet


def main_orientation_axis(dcm_orientation):
    """ Infer orientation from Dicom slices. Returns (0, 'sagittal'), (1, 'coronal'), or (2, 'axial'). """
    # Infer third row of rotation matrix
    main_axis = np.cross(dcm_orientation[0:3], dcm_orientation[3:6])
    main_axis = np.abs(main_axis)
    main_axis = np.argmax(main_axis)

    orientation_from_main_axis = {
        0: 'sagittal',
        1: 'coronal',
        2: 'axial',
    }
    return main_axis, orientation_from_main_axis[main_axis]