import os
import matplotlib.pyplot as plt
import imageprocessing

# Just a couple of utility functions


def savefig(fig, filename, savedir="../fig/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig.savefig(savedir + filename + '.png', bbox_inches='tight', format='png')


def check_image(image, title='', save=False):
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80,
                     facecolor='w', edgecolor='k')
    plt.hist(image.flatten(), bins=128, color='c')
    plt.title(title)
    # plt.xlim((-1000, 2500))
    plt.ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    if save:
        savefig(fig, title, savedir=OUTPUT_FOLDER)


def disp_image(image, sliceindex, title='', save=False):
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80,
                     facecolor='w', edgecolor='k')
    plt.imshow(image[sliceindex, :, :], cmap=plt.cm.bone)
    plt.colorbar()
    if save:
        savefig(fig, title, savedir=OUTPUT_FOLDER)


def disp_image_3axis(image, zf, xf, yf, title='',
                     with_stats=False, save=False):
    (Z, X, Y) = image.shape
    (z, x, y) = map(lambda dim, frac: round(dim * frac), (Z, X, Y), (zf, xf, yf))
    fig, ax = plt.subplots(1, 3, num=None, figsize=(
        16, 12.8), dpi=80, facecolor='w', edgecolor='k')
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

do_save = False

# Locate data
INPUT_FOLDER = 'Y:\\dsb2017\\stage1\\'  # <--change this to your data folder
OUTPUT_FOLDER = '../fig/' + '170227/'
# Select a subset of patient
list_patients = imageprocessing.select_patients(1, INPUT_FOLDER, 'first')
for i, p in enumerate(list_patients):
    # Load a patient
    print(p)
    image, scan = imageprocessing.load_scan(INPUT_FOLDER + p)
    # Preprocess the scan
    image = imageprocessing.preprocess_scan(
        image, scan, do_resample=False,
        do_normalize=True, do_zerocenter=True)
    #Check and display
    # check_image(image,'hist_' + p , save=do_save)
    # disp_image(image, image.shape[0]/2, str(p)+'_orig_s'+str(image.shape[0]/2), save=do_save)
    # Extract lungs ala arnavjain
    bimage = imageprocessing.extract_lungs_in_scan(
        image, return_mask=False, method='arnavjain')
    #Check and display
    check_image(bimage, 'hist_seg_arnavjain_' + p, save=do_save)
    # disp_image(bimage, bimage.shape[0]/2, str(p)+'_arnavjain_s'+str(bimage.shape[0]/2), save=do_save)
    disp_image_3axis(bimage, 0.5, 0.5, 0.25)


if not do_save:
    plt.show()
