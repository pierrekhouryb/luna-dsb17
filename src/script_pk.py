import os
import matplotlib.pyplot as plt
import imageprocessing

def savefig(fig, filename, savedir="./fig/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig.savefig(savedir+filename,bbox_inches='tight')

def check_image(image, title='', save=False):
    fig = plt.figure(num=None, figsize=(16,12.8), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(image.flatten(), bins=128, color='c')
    plt.title(title)
    plt.xlim((-1000, 2500))
    plt.ticklabel_format(style='sci',scilimits=(0,1),axis='y')
    if save:
        savefig(fig, title)
    else:
        plt.draw()

# INPUT_FOLDER = 'Y:\\dsb2017\\sample_images\\'
# list_patients = os.listdir(INPUT_FOLDER)
# list_patients.sort()

INPUT_FOLDER = 'Y:\\dsb2017\\stage1\\'

list_patients = imageprocessing.select_patients(3, INPUT_FOLDER, 'first')
# list_patients = dataio.select_patients_by_index([0, 1, 4, 6], INPUT_FOLDER)

# for i, p in enumerate(list_patients):
#     image, scan = dataio.load_scan(INPUT_FOLDER+patients[i])
#     image = dataio.preprocess_scan(image, scan)
#     check_image(image,patients[i], save=True)
