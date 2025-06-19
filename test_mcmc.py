import os
import emcee
import h5py
import seaborn as sns
import matplotlib.patches as patches
import statistics

import numpy as np
import diversipy as dp
import pandas as pd

from GPErks.log.logger import get_logger
from GPErks.utils.random import set_seed

from time import process_time
from multiprocessing import Pool


import torch
import gpytorch
from GPErks.serialization.path import posix_path
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import load_experiment_from_config_file
from GPErks.train.emulator import GPEmulator
from GPErks.perks.cross_validation import KFoldCrossValidation
from GPErks.train.early_stop import GLEarlyStoppingCriterion
from Historia.shared.design_utils import read_labels

from GPErks_library.mcmc_functions import *
from GPErks.perks.history_matching import Wave


import matplotlib.pyplot as plt
from matplotlib import cm
import corner
import json

def normalise_data(theta, emulator):
    
    x_data = emulator.experiment.dataset.y_train

    norm_arr = []
    
    for col in range(0,x_data.shape[1]):
    
        diff_arr = max(x_data[:,col]) - min(x_data[:,col])

        norm_input_value = ((theta[col] - min(x_data[:,col]))/diff_arr)
        norm_arr.append(norm_input_value)
    
    norm_arr = np.reshape(norm_arr, (1,len(norm_arr)))
    print(norm_arr)
    x_df = pd.DataFrame(data = norm_arr)
    return x_df


    # norm_arr = []
        
    # y_data = emulator.experiment.dataset.y_train

    # norm_output_value = (theta-np.mean(y_data))/(np.std(y_data))
    
    # # diff_arr = max(y_data) - min(y_data)
    # # # print(output_val)
    # # norm_output_value = (((theta - min(y_data))*diff)/diff_arr) + t_min
    # norm_arr.append(norm_output_value)
        
    # norm_arr = np.reshape(norm_arr, (1,len(norm_arr)))
    # y_df = pd.DataFrame(data = norm_arr)

    # return y_df


    ### -------------------------------------------------------------------
    

case_id = '0001'
basefolder = './p' + case_id + '_ani/'

data_outpath = "/home/tmb119/Dropboxportugal_/data/P-" + case_id +"/FinalMesh/"

waveno = 1

datapath = basefolder + 'data/wave'+ str(waveno) + '/'
hmpath = basefolder + 'hm_output/wave' + str(waveno) + '/'

paramfolder = "./parfiles/"

json_settings = paramfolder + "parameter_ranges_" + case_id + ".json"
f_input = open(json_settings,"r")
settings = json.load(f_input)
f_input.close()
# print(settings)
parameters = list(settings.keys())

boundsMaxMin = [(settings[parameter]["lower_limit"], settings[parameter]["upper_limit"]) for parameter in parameters]

mcmc_outpath = basefolder + 'mcmc_output/wave' + str(waveno) + '_v1/'

print('loading mcmc samples...')
filename = mcmc_outpath + "samples_NROY.h5"
reader = emcee.backends.HDFBackend(filename)

burnin = 10000
thin = 10

xlabels = read_labels(datapath + "xlabels_latex.txt")
ylabels = read_labels(datapath + "ylabels_latex.txt")
ndim = len(xlabels)

flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
print(flat_samples.shape)


paramfolder = "./parfiles/"

matchpath = basefolder + 'observed/'

gpepath = basefolder + 'output/wave' + str(waveno) + '/'

exp_mean = np.loadtxt(matchpath + "exp_mean.txt", dtype=float)
exp_std = np.loadtxt(matchpath + "exp_std.txt", dtype=float)

gpe_mode='full'
features_list_file = datapath+"features_idx_list_hm.txt"

active_idx = np.loadtxt(features_list_file,dtype=int)

if active_idx.size>1:
    ylabels = [ylabels[idx] for idx in active_idx]
else:
    ylabels = [ylabels[active_idx]]

# print('History matching using features ')
# for ylab in ylabels:
#     print(ylab)

# ----------------------------------------------------------------
# Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature to match
emulator = []

if active_idx.size>1:
    for idx in active_idx:
        if gpe_mode=="full":
            loadpath = gpepath + str(idx) + "/"
            X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
            y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)

        elif gpe_mode=="best":
            loadpath = gpepath +  str(idx) + "/best_split/"
            X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
            y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
        
        #LOADING GPE 
        # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)
        snapshotpath = loadpath + "snapshot/"
        config_file = loadpath + "snapshot/emulator" + ".ini"
        
        dataset = Dataset(
                X_train,
                y_train,
                x_labels=xlabels,
                y_label=ylabels[idx]
                )
            
        experiment = load_experiment_from_config_file(
        config_file,
        dataset  # notice that we still need to provide the dataset used!
        )
        
        device = "cpu"
        emul = GPEmulator(experiment, device)

        best_model_file = os.readlink(snapshotpath + "best_model.pth")
        
        emul.load_state(best_model_file)

        emulator.append(emul)       

else:
    if gpe_mode=="full":
        loadpath = gpepath + str(active_idx) + "/"
        X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
        y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
    
    elif gpe_mode=="best":
        loadpath = gpepath +  str(active_idx) + "/best_split/"
        X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
        y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
    

    snapshotpath = loadpath + "snapshot/"
    config_file = loadpath + "snapshot/emulator" + ".ini"
    
    dataset = Dataset(
            X_train,
            y_train,
            x_labels=xlabels,
            y_label=ylabels
            )
    
    #LOADING GPE 
    # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)

    experiment = load_experiment_from_config_file(
    config_file,
    dataset  # notice that we still need to provide the dataset used!
    )

    # print(experiment.model)
    # print(experiment.likelihood)
    
    device = "cpu"
    emul = GPEmulator(experiment, device)
    # print('**********************************')
    # print(emul.experiment.model)
    # print(emul.experiment.likelihood)

    best_model_file = os.readlink(snapshotpath + "best_model.pth")
    
    emul.load_state(best_model_file)

    # print('**********************************')
    # print(emul.experiment.model)
    # print(emul.experiment.likelihood)

    # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)
    emulator.append(emul)  

# W = Wave()
# W.load(hmpath+"/wave_"+str(waveno)+ ".json")
# # print(W.NIMP)

# W_initial = Wave()
# W_initial.load(basefolder + "hm_output/wave1/wave_1.json")

# X_total = W_initial.IMP
# X_total =np.vstack((W_initial.NIMP, W_initial.IMP))

# print(np.shape(X_total))
# print(X_total)

errs = []
# print(emulator)
figure_size = (16.0,8.0)
fig = plt.figure(figsize = figure_size)
afont = {'fontname':'Arial'}
# print(flat_samples)
for i, emul in enumerate(emulator):
    y = emul.predict(flat_samples)
    y_mode = statistics.mode(y[0])
    # print(y_mode)
    ax = plt.subplot(2, 4, i+1)
# print((y[0]))
    ax = sns.histplot(y[0],color=cm.RdPu(0.48),alpha=1, zorder=100,edgecolor='none', linewidth=0)
    ax.axvline(exp_mean[i],color=cm.Blues(0.95), lw=1, label="target", alpha=1,zorder=200,linestyle='--')
    ax.add_patch(
        patches.Rectangle(
        xy=(exp_mean[i]-(2*exp_std[i]), 0),  # point of origin.
        width=2*exp_std[i], height=5000, linewidth=1,
        color='#B0BEC5', fill=True, alpha=0.6))

    ax.add_patch(
        patches.Rectangle(
        xy=(exp_mean[i], 0),  # point of origin.
        width=2*exp_std[i], height=5000, linewidth=1,
        color='#B0BEC5', fill=True, alpha=0.6))

    ax.set_xlabel(ylabels[i],**afont, fontsize=16, labelpad=1)
    ax.set_ylabel("Count",**afont, fontsize=16)

    # y_prior = emul.predict(X_total)
    # y_prior_mode = statistics.mode(y_prior[0])
    # ax = sns.histplot(y_prior[0],color=cm.Blues(0.30),alpha=1,zorder=0,edgecolor='none', linewidth=0)

    

    ax.set_ylim(0,3500)
    # if i < 6:
    #     ax.set_xlim(1,9)

    ax.tick_params(left = False, labelleft = False,
                bottom = True, labelbottom = True)
    ax.tick_params(axis='x', which='major', labelsize=14)



    # ax.legend()

    err = exp_mean[i] - y_mode
    errs.append(err)




plt.savefig(mcmc_outpath + "test_distribution_" + case_id +".png", dpi=300)
# plt.show()


np.savetxt(mcmc_outpath + 'posterior_errors.txt', errs)



# for i, emul in enumerate(emulator[:1]):
#     print("emulator" + str(i) +"_X_train")
#     print(emul.experiment.scaled_data.X_train)
#     print(emul.experiment.scaled_data.X_train[0])
#     # print(emul.experiment.dataset.X_train)

#     theta = emul.experiment.dataset.X_train[0]
#     print(theta)
#     inputNorm = normalise_data(theta=theta, emulator=emul)

#     print(inputNorm)

