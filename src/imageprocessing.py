#!/usr/bin/env python

""" Comments about the file
"""

import logging
import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import random
from skimage import data, filters, measure, morphology, feature, segmentation
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# Just a couple of utility functions
def savefig(fig, filename, savedir="../fig/"):
    savedir = os.path.abspath(savedir)
    if not os.path.exists(savedir):
        logger.warning('Out dir not existing, creating {}'.format(savedir))
        os.makedirs(savedir)
    logger.info(' - save image to: {}'.format(savedir))
    fig.savefig(os.path.join(savedir, filename + '.png'),
                bbox_inches='tight', format='png')


def check_image(image, title='', save=False, output_dir=None):
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80,
        facecolor='w', edgecolor='k')
    plt.hist(image.flatten(), bins=128, color='c')
    plt.title(title)
    # plt.xlim((-1000, 2500))
    plt.ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    if save:
        savefig(fig, title, savedir=output_dir)
    else:
        plt.show()


def disp_image(image, sliceindex, title='', save=False, output_dir=None):
    if isinstance(sliceindex, float):
        logger.warning(
            'sliceindex is a float: {}, rounding to: {}'.format(
                sliceindex, int(sliceindex)))
        sliceindex = int(sliceindex)
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80,
                     facecolor='w', edgecolor='k')
    plt.imshow(image[sliceindex, :, :], cmap=plt.cm.bone)
    plt.colorbar()
    if save:
        savefig(fig, title, savedir=output_dir)
    else:
        plt.show()


def disp_image_3axis(image, zf, xf, yf, title='', with_stats=False,
                     save=False, output_dir=None):
    (Z, X, Y) = image.shape
    (z, x, y) = map(lambda dim, frac: round(dim * frac), (Z, X, Y), (zf, xf, yf))
    fig, ax = plt.subplots(1, 3, num=None, figsize=(16, 12.8), dpi=80,
                           facecolor='w', edgecolor='k')
    ax.ravel()[0].imshow(image[z, :, :], cmap=plt.cm.bone)
    ax[0].plot([y, y], [0, X], 'g-')
    ax[0].plot([0, Y], [x, x], 'g-')
    ax[0].set_xlim([0, X])
    ax[0].set_ylim([0, Y])
    ax[0].set_title('z=' + str(z))
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].imshow(image[:, x, :], cmap=plt.cm.bone)
    ax[1].plot([y, y], [0, Z], 'g-')
    ax[1].plot([0, Y], [z, z], 'g-')
    ax[1].set_xlim([0, Y])
    ax[1].set_ylim([0, Z])
    ax[1].set_title('x=' + str(x))
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('z')
    ax[2].imshow(image[:, :, y], cmap=plt.cm.bone)
    ax[2].plot([x, x], [0, Z], 'g-')
    ax[2].plot([0, X], [z, z], 'g-')
    ax[2].set_xlim([0, X])
    ax[2].set_ylim([0, Z])
    ax[2].set_title('y=' + str(y))
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('z')
    if save:
        savefig(fig, title, savedir=output_dir)
    else:
        plt.show()


# LUNA 16 constants
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25


def select_patients(n, folder, method='first'):
    """
    Select n patients from folder, based on method.
    Folder must be containing patients folders/files only.
    """
    patients = sorted(os.listdir(folder))
    if not os.path.isdir(os.path.join(folder, patients[0])):
        #if the dataset is not dsb2017 but LUNA16, we only list the mhd files as patients
        patients = [p for p in patients if os.path.splitext(p)[-1] == '.mhd']

    if not patients:
        logger.warning('No patients found.')
        return

    if len(patients) < n:
        logger.warning(
            'Not enough patient s in this folder. Found {} but need {}.'.format(
                len(patients), n))
        return

    if (n == -1) or (n > len(patients)):
        return patients

    if method == 'first':
        return patients[:n]

    if method == 'last':
        return patients[-n:]

    if method == 'random':
        out_l = []
        while n > 0:
            idx = random.randrange(0, len(patients))
            out_l.append(patients[idx])
            patients.remove(patients[idx])
            n -= 1
        return out_l


def select_patients_by_index(indices, folder):
    """
    Select patients from folder based on their index.
    Folder must be containing patients folders only.
    """
    patients = sorted(os.listdir(folder))
    if not os.path.isdir(os.path.join(folder, patients[0])):
        #if the dataset is not dsb2017 but LUNA16, we only list the mhd files as patients
        patients = [p for p in patients if os.path.splitext(p)[-1] == '.mhd']

    if not patients:
        print('No patients found.')
        return

    if len(patients) < max(indices):
        logger.warning(
            'Not enough patients in this folder. Found {} but need {}.'.format(
                len(patients),
                max(indices)))
        return

    return [patients[i] for i in indices]


def load_scan(patient_folder_or_file):
    """
    Load single patient.
    Returns the CT scan in Hounsfield Units as a 3D numpy array with (Z,X,Y),
    a dictionary with few meta info and the raw data.
    """
    def load_dicom(patient_folder):
        metainfo = {}
        raw_dicom = [dicom.read_file(
            os.path.abspath(os.path.join(patient_folder, s))
        ) for s in os.listdir(patient_folder)]
        raw_dicom.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(
                raw_dicom[0].ImagePositionPatient[2] -
                raw_dicom[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                raw_dicom[0].SliceLocation -
                raw_dicom[1].SliceLocation)

        for s in raw_dicom:
            s.SliceThickness = slice_thickness

        metainfo["spacing"] = np.array(list(map(float, ([raw_dicom[0].SliceThickness] + raw_dicom[0].PixelSpacing))))

        def get_pixels_hu(raw_dicom):
            image = np.stack([s.pixel_array for s in raw_dicom])
            # Convert to int16 (from sometimes int16),
            # should be possible as values should always be low enough (<32k)
            image = image.astype(np.int16)

            # Set outside-of-scan pixels to 0
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            # Convert to Hounsfield units (HU)
            for slice_number in range(len(raw_dicom)):

                intercept = raw_dicom[slice_number].RescaleIntercept
                slope = raw_dicom[slice_number].RescaleSlope

                if slope != 1:
                    image[slice_number] = slope * \
                        image[slice_number].astype(np.float64)
                    image[slice_number] = image[slice_number].astype(np.int16)

                image[slice_number] += np.int16(intercept)

            return np.array(image, dtype=np.int16)

        return (get_pixels_hu(raw_dicom), metainfo, raw_dicom)

    def load_mhd(patient_file):
        # Reads the image using SimpleITK
        raw_itk = sitk.ReadImage(patient_file)
        image = sitk.GetArrayFromImage(raw_itk)

        # Not sure why, background pixels are set to -3200 in the LUNA16
        # datasets. Setting intercept to -1024 instead to match dsb2017.
        image[image < -1024] = -1024

        metainfo = {}
        metainfo["spacing"] = np.array(list(reversed(raw_itk.GetSpacing())))
        metainfo["origin"] = np.array(list(reversed(raw_itk.GetOrigin())))

        return (image, metainfo, raw_itk)

    # Naive check if input is a dicom folder or an mhd file.
    if(os.path.isdir(patient_folder_or_file)):
        return load_dicom(patient_folder=patient_folder_or_file)
    else:
        return load_mhd(patient_file=patient_folder_or_file)

def preprocess_scan(image, metainfo, do_resample=True, do_normalize=True,
        do_zerocenter=True):
    """
    Preprocess single patient.
    """
    logger.info(' - Preprocess the scan')
    if not (do_resample or do_normalize or do_zerocenter):
        return image

    def resample(image, spacing, new_spacing=[1, 1, 1]):
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(
            image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def normalize(image):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    def zerocenter(image):
        image = image - PIXEL_MEAN
        return image

    if do_resample:
        preprocessed, new_spacing = resample(image, metainfo["spacing"])
        metainfo["new_spacing"] = new_spacing
    if do_normalize:
        preprocessed = image if not do_resample else preprocessed
        preprocessed = normalize(preprocessed)
    if do_zerocenter:
        preprocessed = image if (
            not do_resample and not do_normalize) else preprocessed
        preprocessed = zerocenter(preprocessed)

    return preprocessed, metainfo


def extract_lungs_in_scan(in_image, return_mask=False, method='arnavjain'):
    """
    Extract lungs from 3d scan data based on method.
    """
    logger.info(' - extract lungs in scan ({})'.format(method))

    def extract_lungs_in_scan_arnavjain(in_image):
        threshold = (-400 - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN

        def segment_slice(im):
            binary = im < threshold
            cleared = segmentation.clear_border(binary)
            label_image = measure.label(cleared)

            areas = sorted([r.area for r in measure.regionprops(label_image)])
            if len(areas) > 2:
                for region in measure.regionprops(label_image):
                    if region.area < areas[-2]:
                        for coordinates in region.coords:
                            label_image[coordinates[0], coordinates[1]] = 0
            binary = label_image > 0
            selem = morphology.disk(2)
            binary = morphology.binary_erosion(binary, selem)
            selem = morphology.disk(10)
            binary = morphology.binary_closing(binary, selem)

            edges = filters.roberts(binary)
            binary = scipy.ndimage.binary_fill_holes(edges)

            return binary

        out_mask = np.zeros(in_image.shape)
        for i in range(in_image.shape[0]):
            out_mask[i, ...] = segment_slice(in_image[i, ...])

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
        binary_image = np.array(in_image > threshold, dtype=np.int8) + 1
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_label = labels[0, 0, 0]

        # Fill the air around the person
        binary_image[background_label == labels] = 2

        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image

    if method == 'arnavjain':
        seg_method = extract_lungs_in_scan_arnavjain
    elif method == 'zuidhof':
        seg_method = extract_lungs_in_scan_zuidhof

    mask = seg_method(in_image)

    if return_mask:
        return mask
    else:
        return in_image * mask
