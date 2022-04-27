"""
Compute_metrics.py
Based on *Evaluate.compute_mean_metrics*
For a given model name, it creates a .csv file with summary statistics:
    Median, MAD, Mean, SD
For the sound-to-distortion (SDR) ratio in test files of MUSDB18 dataset,
located at json_folder
"""
import numpy as np
import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model_name = "testing_mhe_9311" # change model name here
filename = str(model_name + '.csv')
out_path = r"D:\Users\kijam\Documents\3CollegeClasses\CS534\OurCodebase\Wave-U-Net\Results\SDR" # where to save SDR metrics

json_folder = r"D:\Users\kijam\Documents\3CollegeClasses\CS534\OurCodebase\Wave-U-Net\SourceEstimates\test" # add "r" before the path
compute_averages = True
plot_hist = False
metric = "SDR"

files = glob.glob(os.path.join(json_folder, "*.json"))
inst_list = None
print("Found " + str(len(files)) + " JSON files to evaluate...")
for path in files:
    #print(path)
    if path.__contains__("test.json"):
        print("Found test JSON, skipping...")
        continue

    with open(path, "r") as f:
        js = json.load(f)

    if inst_list is None:
        inst_list = [list() for _ in range(len(js["targets"]))]

    for i in range(len(js["targets"])):
        inst_list[i].extend([np.float(f['metrics'][metric]) for f in js["targets"][i]["frames"]])

#return np.array(sdr_acc), np.array(sdr_voc)
inst_list = [np.array(perf) for perf in inst_list]

if compute_averages:
    out = [(np.nanmedian(perf), np.nanmedian(np.abs(perf - np.nanmedian(perf))), np.nanmean(perf), np.nanstd(perf)) for perf in inst_list]
    # convert to pandas DF
    df_out = pd.DataFrame(out, columns=['Med','MAD','Mean','SD'], index=['voice','acc'])
    # save results to folder
    df_out.to_csv(os.path.join(out_path, filename))
else:
    # convert to pandas DF
    df_out_full = pd.DataFrame(inst_list, index=['voice','acc'])
    # transpose DF
    df_out_full = df_out_full.transpose(copy=True)
    # save results to folder
    filename_full = str('full_data_' + filename)
    df_out_full.to_csv(os.path.join(out_path, filename_full))
    # drop NA
    df_out_full = df_out_full.dropna()

    if plot_hist:
        # plot histograms    
        sns.distplot(df_out_full['voice'], color='#9b59b6', label='voice', hist=False, kde=True)
        sns.distplot(df_out_full['acc'], color='#3498db', label='accompaniment', hist=False, kde=True)
        plt.legend(prop={'size': 12})
        plt.suptitle('SDR distributions for estimated voice and accompaniment')
        plt.title('Baseline') # CHANGE TITLE HERE
        plt.rc('text', usetex=True) # import latex extension
        plt.rc('font', family='serif') # use latex font
        plt.xlabel('dB')
        plt.ylabel('Frequency')