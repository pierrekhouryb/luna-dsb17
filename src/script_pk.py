import os
import matplotlib.pyplot as plt
import imageprocessing

# Just a couple of utility functions
def savefig(fig, filename, savedir="../fig/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig.savefig(savedir+filename,bbox_inches='tight')

def check_image(image, title='', save=False):
    fig = plt.figure(num=None, figsize=(16,12.8), dpi=80, facecolor='w', edgecolor='k')
    plt.hist(image.flatten(), bins=128, color='c')
    plt.title(title)
    # plt.xlim((-1000, 2500))
    plt.ticklabel_format(style='sci',scilimits=(0,1),axis='y')
    if save:
        savefig(fig, title)

def disp_image(image, sliceindex, title='', save=False):
    fig = plt.figure(num=None, figsize=(16,12.8), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(image[sliceindex,:,:], cmap=plt.cm.bone)
    plt.colorbar()
    if save:
        savefig(fig, title)

do_save = True

#Locate data
INPUT_FOLDER = 'Y:\\dsb2017\\stage1\\' #<--change this to your data folder
#Select a subset of patient
list_patients = imageprocessing.select_patients(15, INPUT_FOLDER, 'first')
for i, p in enumerate(list_patients):
    #Load a patient
    image, scan = imageprocessing.load_scan(INPUT_FOLDER+p)
    #Preprocess the scan
    image = imageprocessing.preprocess_scan(image, scan)
    #Check and display
    check_image(image,'hist_' + p , save=do_save)
    disp_image(image, image.shape[0]/2, str(p)+'_orig_s'+str(image.shape[0]/2), save=do_save)
    #Extract lungs ala arnavjain
    bimage = imageprocessing.extract_lungs_in_scan(image, return_mask=False, method='arnavjain')
    #Check and display
    check_image(bimage,'hist_seg_arnavjain_' + p, save=do_save)
    disp_image(bimage, bimage.shape[0]/2, str(p)+'_arnavjain_s'+str(bimage.shape[0]/2), save=do_save)
    #Extract lungs ala zuidhof
    bimage = imageprocessing.extract_lungs_in_scan(image, return_mask=False, method='zuidhof')
    #Check and display
    check_image(bimage,'hist_seg_zuidhof_' + p, save=do_save)
    disp_image(bimage, bimage.shape[0]/2, str(p)+'_zuidhof_s'+str(bimage.shape[0]/2), save=do_save)
    print(i)

if not do_save:
    plt.show()
