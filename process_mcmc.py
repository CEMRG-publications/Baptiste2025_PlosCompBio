import os
import emcee
import h5py

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

from mcmc_functions import *

import matplotlib.pyplot as plt
from matplotlib import cm
import corner
import json


case_id = '0005'

basefolder = './p' + case_id + '_ani/'

# data_outpath = "/home/tmb119/Dropbox/data/CTCRT" + case_id +"/FinalMesh/"

waveno = 1

datapath = basefolder + 'data/wave'+ str(waveno) + '/'

anal_type = "NROY"

paramfolder = "./parfiles/"

json_settings = paramfolder + "parameter_ranges_" + case_id + ".json"
f_input = open(json_settings,"r")
settings = json.load(f_input)
f_input.close()
# print(settings)
parameters = list(settings.keys())

boundsMaxMin = [(settings[parameter]["lower_limit"], settings[parameter]["upper_limit"]) for parameter in parameters]

mcmc_outpath = basefolder + 'mcmc_output/wave' + str(waveno) + '/'

print('loading mcmc samples...')
if anal_type == "whole_space":
    filename = mcmc_outpath + "samples.h5"
elif anal_type == "NROY":
    filename = mcmc_outpath + "samples_NROY.h5"
else:
    print("check analysis type specified: whole_space OR NROY")


reader = emcee.backends.HDFBackend(filename)

burnin = 210000
thin = 10

xlabels = read_labels(datapath + "xlabels_latex.txt")
ylabels = read_labels(datapath + "ylabels.txt")
ndim = len(xlabels)

fig, axes = plt.subplots(ndim, figsize=(15, 10), sharex=True)

samples = reader.get_chain()
print(np.shape(samples))

print('number of samples: ', len(samples))
labels = xlabels
ndim = len(xlabels)

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], color=cm.RdPu(0.95), alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i],rotation=0)
    ax.yaxis.set_label_coords(-0.1, 0.2)

axes[-1].set_xlabel("step number");


plt.savefig(mcmc_outpath + 'samples_' + anal_type +'.png', dpi=300)


flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
print(flat_samples.shape)

outputs = []

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

max_log_prob = np.max(log_prob_samples)

print('prob: ', max_log_prob)

# outputs.append(max_log_prob)

max_log_prob_idx = np.where(log_prob_samples == max_log_prob)[0]
print('idx: ' ,max_log_prob_idx)

sorted_indices = np.argsort(log_prob_samples, axis=0)
max_prob_idx = sorted_indices[-50:]
max_prob_idx = max_prob_idx[::-1]

max_prob_values = flat_samples[max_log_prob_idx][0]

print(max_prob_values)

test_values = np.take(flat_samples,max_prob_idx,axis=0)


np.savetxt(mcmc_outpath + 'max_prob_values_' + anal_type + '.txt', test_values)

avg = np.mean(flat_samples, axis=0)
std = np.std(flat_samples, axis=0)

out_dict = {}

regions =  np.array(["anterior", "posterior", "septum", "lateral", "roof"])

for i,region in enumerate(regions):
    out_dict["stiffness_" + region] = max_prob_values[i]

with open(mcmc_outpath + "/calibrated_stiffness_" + anal_type + ".json", "w") as outfile: 
    print(outfile)
    json.dump(out_dict, outfile, indent=4)

# print(max_prob_values)


fig = corner.corner(
     flat_samples, labels=xlabels, label_kwargs=dict(fontsize=24),color=cm.RdPu(0.95)
);
corner.overplot_lines(fig, max_prob_values, color=cm.Blues(0.50), lw=5, alpha=0.75)
corner.overplot_points(fig, max_prob_values[None], marker="s", color=cm.Blues(0.50), markersize=5, alpha=0.75)

# Extract the axes

axes = np.array(fig.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]
    ax.set_xlim(boundsMaxMin[i])
    ax.set_yticks([])
    ax.set_xticks([])    
    

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.set_xlim(boundsMaxMin[xi])
        ax.set_ylim(boundsMaxMin[yi])
        ax.set_yticks([])
        ax.set_xticks([]) 

plt.savefig(mcmc_outpath + 'cornerplot_' + anal_type +'.png', dpi=300)

fig = corner.corner(
     flat_samples, labels=xlabels, label_kwargs=dict(fontsize=18),color='k'
);

for row in range(50):
    
    corner.overplot_lines(fig, test_values[row], color=cm.RdPu(0.95), lw=3, alpha=0.5)
    corner.overplot_points(fig, test_values[row][None], marker="s", color=cm.RdPu(0.95), markersize=5)

# Extract the axes

axes = np.array(fig.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]
    ax.set_xlim(boundsMaxMin[i])

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.set_xlim(boundsMaxMin[xi])
        ax.set_ylim(boundsMaxMin[yi])

plt.savefig(mcmc_outpath + 'cornerplot_w_testvalues_' + anal_type +'.png', dpi=300)




