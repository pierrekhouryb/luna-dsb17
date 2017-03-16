import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import data, filters, measure, morphology, feature, segmentation

# LUNA 16 constants
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

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

# Load single patient.
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

# Preprocess single patient.
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
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    def zerocenter(image):
        image = image - PIXEL_MEAN
        return image

    if do_resample:
        preprocessed = resample(image, scan)
    if do_normalize:
        preprocessed = image if not do_resample else preprocessed
        preprocessed = normalize(preprocessed)
    if do_zerocenter:
        preprocessed = image if (not do_resample and not do_normalize) else preprocessed
        preprocessed = zerocenter(preprocessed)

    return preprocessed

# Extract lungs from 3d scan data based on method.
def extract_lungs_in_scan(in_image, return_mask=False, method='arnavjain'):
    def extract_lungs_in_scan_arnavjain(in_image):
        threshold = (-400 - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
        def segment_slice(im):
            binary = im < threshold
            cleared = segmentation.clear_border(binary)
            label_image = measure.label(cleared)

            areas = [r.area for r in measure.regionprops(label_image)]
            areas.sort()
            if len(areas) > 2:
                for region in measure.regionprops(label_image):
                    if region.area < areas[-2]:
                        for coordinates in region.coords:
                               label_image[coordinates[0], coordinates[1]] = 0
            binary = label_image > 0
            selem = morphology.disk(2)
            binary =  morphology.binary_erosion(binary, selem)
            selem =  morphology.disk(10)
            binary =  morphology.binary_closing(binary, selem)

            edges = filters.roberts(binary)
            binary = scipy.ndimage.binary_fill_holes(edges)

            return binary

        out_mask = np.zeros(in_image.shape)
        for i in range(in_image.shape[0]):
            out_mask[i,...] = segment_slice(in_image[i,...])

        return out_mask

    def extract_lungs_in_scan_zuidhof(in_image, fill_lung_structures=True):
        def largest_label_volume(im, bg=-1):
            vals, counts = np.unique(im, return_counts=True)

            counts = counts[vals != bg]
            vals = vals[vals != bg]

            if len(counts) > 0:
                return vals[np.argmax(counts)]
            else:
                return None
        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        threshold = (-400 - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
        binary_image = np.array(in_image > threshold, dtype=np.int8)+1
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0,0,0]

        #Fill the air around the person
        binary_image[background_label == labels] = 2


        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = largest_label_volume(labeling, bg=0)

                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1


        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image

    if method=='arnavjain':
        seg_method = extract_lungs_in_scan_arnavjain
    elif method=='zuidhof':
        seg_method = extract_lungs_in_scan_zuidhof

    mask = seg_method(in_image)

    if return_mask:
        return mask
    else:
        return in_image*mask
