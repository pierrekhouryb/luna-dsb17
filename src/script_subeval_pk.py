import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def list_submissions(folder):
    submissions = glob.glob(folder+"\\*.csv")
    return submissions

def list_last_submissions(folder, k=3):
    all_subs = list_submissions(folder)
    return all_subs[-k:]

def read_submissions(list_of_files):
    return dict((os.path.basename(f), pd.read_csv(f)) for f in list_of_files)

def plot_subm_histo(name, data, norm=True):
    d = data["cancer"]
    if norm:
        d = (d-np.min(d)) / (np.max(d)-np.min(d))

    plt.plot(d, label=name)
    plt.legend()

subm_folder = "..\\reports"
subm_files = list_last_submissions(subm_folder, 3)
subms_dict = read_submissions(subm_files)

for subm in subms_dict:
    plot_subm_histo(subm, subms_dict[subm])

plt.show()
