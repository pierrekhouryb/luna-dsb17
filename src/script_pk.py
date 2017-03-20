#!/usr/bin/env python

""" Comments about the file
"""

import os
import logging
import matplotlib.pyplot as plt
import imageprocessing
import argparse

logger = logging.getLogger(__name__)


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
        logger.info('output dir set to' + out_dir)
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
    list_patients = imageprocessing.select_patients(15, INPUT_FOLDER, 'random')
    for i, p in enumerate(list_patients):
        logger.info('Load patient {}/{} ({})'.format(i + 1, len(list_patients),
                                                     p))

        image, meta, scan = imageprocessing.load_scan(os.path.join(INPUT_FOLDER, p))
        imageprocessing.check_image(image, str(p) + '_hist_orig', save=do_save,
                        output_dir=OUTPUT_FOLDER)
        # Preprocess the scan
        image, meta = imageprocessing.preprocess_scan(
            image, meta, do_resample=True, do_normalize=True, do_zerocenter=True)
        # Check and display
        # check_image(image, 'hist_' + p, save=do_save, output_dir=OUTPUT_FOLDER)
        # disp_image(image, image.shape[0] / 2,
        #           str(p) + '_orig_s' + str(image.shape[0] / 2), save=do_save)
        # Extract lungs ala arnavjain
        imageprocessing.disp_image_3axis(image, 0.5, 0.5, 0.25,
                    str(p) + '_preprocessed_s' + str(int(image.shape[0] / 2)),
                    save=do_save, output_dir=OUTPUT_FOLDER)
        bimage = imageprocessing.extract_lungs_in_scan(
            image, return_mask=False, method='arnavjain')
        # Check and display
        imageprocessing.check_image(bimage, str(p) + '_hist_seg_arnavjain', save=do_save,
                    output_dir=OUTPUT_FOLDER)
        imageprocessing.disp_image(bimage, bimage.shape[0] / 2,
                   str(p) + '_arnavjain_s' + str(int(bimage.shape[0] / 2)),
                   save=do_save, output_dir=OUTPUT_FOLDER)
        imageprocessing.disp_image_3axis(bimage, 0.5, 0.5, 0.25,
                    str(p) + '_arnavjain_s' + str(int(bimage.shape[0] / 2)),
                    save=do_save, output_dir=OUTPUT_FOLDER)

        logger.info('Processing patient {} done'.format(i + 1))


if __name__ == "__main__":
    main()
