import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology


# Select n patients from folder, based on method.
# Folder must be containing patients folders only.
def select_patients(n, folder, method='first'):
    patients = os.listdir(folder)
    patients.sort()

    if not patients:
        print('No patients found.')
        return

    if len(patients) < n:
        print('Not enough patient s in this folder. Found {} but need {}.'.format(len(patients), n))
        return

    if n==-1:
        return patients

    if method=='first':
        return patients[:n]

    if method=='last':
        return patients[-n:]

    if method=='random': # to implement
        return 0

# Select patients from folder based on their index.
# Folder must be containing patients folders only.
def select_patients_by_index(indices, folder):
    patients = os.listdir(folder)
    patients.sort()

    if not patients:
        print('No patients found.')
        return

    if len(patients) < max(indices):
        print('Not enough patients in this folder. Found {} but need {}.'.format(len(patients), max(indices)))
        return

    return [patients[i] for i in indices]

# Load single patient
def load_scan(patient_folder):
    slices = [dicom.read_file(patient_folder + '/' + s) for s in os.listdir(patient_folder)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    return (get_pixels_hu(slices), slices)

# Preprocess single patient
def preprocess_scan(image, scan, do_resample=True, do_normalize=True, do_zerocenter=True):
    if not (do_resample or do_normalize or do_zerocenter):
        return image

    def resample(image, scan, new_spacing=[1,1,1]):
        # Determine current pixel spacing
        spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image #, new_spacing

    def normalize(image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    def zerocenter(image):
        PIXEL_MEAN = 0.25
        image = image - PIXEL_MEAN
        return image

    if do_resample:
        preprocessed = resample(image, scan)
    if do_normalize:
        preprocessed = normalize(image)
    if do_zerocenter:
        preprocessed = zerocenter(image)

    return preprocessed
