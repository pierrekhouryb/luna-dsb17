#!/usr/bin/env python

""" Comments about the file
"""

import os
import logging
import matplotlib.pyplot as plt
import imageprocessing
import argparse

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
                     save=False):
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
        savefig(fig, title, savedir=OUTPUT_FOLDER)
    else:
        plt.show()


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_dir', help="input directory")
    p.add_argument('-s', '--save_output', help="Save the results")
    p.add_argument('-o', '--output_dir', help="Output directory")

    args = p.parse_args()

    if (args.input_dir is None) and (
            args.save_output is None) and (args.output_dir is None):
        logger.info('* No arguments, using hardcoded values')
        # Locate data
        in_dir = '../data/in'
        in_dir = os.path.abspath(in_dir)
        out_dir = os.path.abspath(os.path.join(in_dir, '../fig/', '170227/'))
        do_save = False
    else:
        in_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(in_dir):
            raise ValueError('{} is not a valid input path.'.format(in_dir))

        out_dir = os.path.abspath(args.output_dir)
        if not os.path.isdir(out_dir):
            raise ValueError('{} is not a valid output path.'.format(out_dir))

        do_save = args.save_output
        if isinstance(do_save, str):
            if do_save.lower() in ['true']:
                do_save = True
            else:
                do_save = False

    logger.info('Input parameters:')
    logger.info(' - data in: {}'.format(in_dir))
    logger.info(' - data out: {}'.format(out_dir))
    logger.info(' - save results: {}'.format(do_save))

    return in_dir, out_dir, do_save


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)-5s] %(lineno)s | %(message)s',
                        datefmt='%M:%S')

    INPUT_FOLDER, OUTPUT_FOLDER, do_save = parse_arguments()

    # Select a subset of patient
    list_patients = imageprocessing.select_patients(1, INPUT_FOLDER, 'first')
    for i, p in enumerate(list_patients):
        logger.info('Load patient {}/{} ({})'.format(i + 1, len(list_patients),
                                                     p))

        image, scan = imageprocessing.load_scan(os.path.join(INPUT_FOLDER, p))
        # Preprocess the scan
        image = imageprocessing.preprocess_scan(
            image, scan, do_resample=False, do_normalize=True, do_zerocenter=True)
        # Check and display
        # check_image(image, 'hist_' + p, save=do_save, output_dir=OUTPUT_FOLDER)
        # disp_image(image, image.shape[0] / 2,
        #           str(p) + '_orig_s' + str(image.shape[0] / 2), save=do_save)
        # Extract lungs ala arnavjain
        bimage = imageprocessing.extract_lungs_in_scan(
            image, return_mask=False, method='arnavjain')
        # Check and display
        check_image(bimage, 'hist_seg_arnavjain_' + p, save=do_save,
                    output_dir=OUTPUT_FOLDER)
        # disp_image(bimage, bimage.shape[0] / 2,
        #            str(p) + '_arnavjain_s' + str(bimage.shape[0] / 2),
        #            save=do_save)
        disp_image_3axis(bimage, 0.5, 0.5, 0.25)

        logger.info('Processing patient {} done'.format(i + 1))


if __name__ == "__main__":
    main()
