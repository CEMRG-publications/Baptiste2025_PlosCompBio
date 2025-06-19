import os
import emcee
import h5py
import json

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
from GPErks.serialization.labels import read_labels_from_file

from mcmc_functions import *

import matplotlib.pyplot as plt
from matplotlib import cm
import corner

from GPErks.perks.history_matching import Wave
from GPErks.utils.array import get_minmax

# from Historia.history import hm
# from Historia.shared.design_utils import get_minmax


def main():
    device = "cpu"
    devices = [device]  # a list of devices

    case_id = "28"

    basefolder = "./ctcrt" + case_id + "_uniform/"

    waveno = 3

    # option to restrict input space of MCMC analysis to NROY of final wave of HM
    restrict_inputspace = True

    paramfolder = "./parfiles/"

    datapath = basefolder + 'data/wave'+ str(waveno) + '/'

    matchpath = basefolder + 'observed/'

    gpepath = basefolder + 'output/wave' + str(waveno) + '/'

    hmpath = basefolder + 'hm_output/wave' + str(waveno) + '/'

    mcmc_outpath = basefolder + 'mcmc_output/wave' + str(waveno) + '/'

    cmd = 'mkdir -p ' + mcmc_outpath
    os.system(cmd)


    features_list_file = None
    gpe_mode='full'

    # ----------------------------------------------------------------
    # Load input parameters and output features' names
    xlabels = read_labels_from_file(datapath + "xlabels.txt")
    ylabels = read_labels_from_file(datapath + "ylabels.txt")
    features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}


    # ----------------------------------------------------------------
    #### Load experimental values (mean +- std) you aim to match ######

    exp_mean = np.loadtxt(matchpath + "exp_mean.txt", dtype=float)


    exp_std = np.loadtxt(matchpath + "exp_std.txt", dtype=float)
    exp_var = np.power(exp_std, 2)

    print(exp_std)
    print(exp_var)

    # ----------------------------------------------------------------
    if features_list_file is None:
        features_list_file = datapath+"features_idx_list_hm.txt"

    active_idx = np.loadtxt(features_list_file,dtype=int)

    # active_idx = np.array(0)

    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]

    if active_idx.size>1:
        ylabels = [ylabels[idx] for idx in active_idx]
    else:
        ylabels = [ylabels[active_idx]]

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

    print(emulator) 

    ##**********************************************
    if restrict_inputspace == True:

        #loading final wave parameter bounds 

        W = Wave()
        W.load(hmpath+"/wave_"+str(waveno)+ ".json")

        boundsMaxMin = get_minmax(W.NIMP)
        centre = (np.array(boundsMaxMin)[:,1]+np.array(boundsMaxMin)[:,0])/2
        print(boundsMaxMin)
        print(centre)

    ##**********************************************

    elif restrict_inputspace == False:

        #loading input parameter bounds 

        json_settings = paramfolder + "parameter_ranges_" + case_id + ".json"
        f_input = open(json_settings,"r")
        settings = json.load(f_input)
        f_input.close()
        # print(settings)
        parameters = list(settings.keys())

        boundsMaxMin = [(settings[parameter]["lower_limit"], settings[parameter]["upper_limit"]) for parameter in parameters]

        centre = (np.array(boundsMaxMin)[:,1]+np.array(boundsMaxMin)[:,0])/2
        input_range = np.array(boundsMaxMin)[:,1] - np.array(boundsMaxMin)[:,0]
        print(boundsMaxMin)
        print(input_range)
        print(centre)

    ##**********************************************
    ## Initialising walkers

    ndim = len(boundsMaxMin)
    print(ndim)
    nwalkers = 2*ndim # should be at least 2*ndim
    print(nwalkers)                 
    p0 = np.random.multivariate_normal(centre, 0.000000001*np.identity(ndim), size=(nwalkers))

    # p0 = np.random.multivariate_normal(centre,0.000000001*np.diagonal(), size=(nwalkers))

    # y_val = torch.tensor(exp_mean).float()
    y_val=exp_mean
    sigma2 = exp_var


    ##***********************************************
    ## Perform MCMC analysis


    filename = mcmc_outpath + "samples_NROY.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)


    # t1_start = process_time() 
    torch.set_num_threads(nwalkers)

    with Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[emulator,y_val, sigma2, boundsMaxMin], backend=backend)
        sampler.run_mcmc(p0, 100000, progress=True)

    # t1_stop = process_time() 

    # print("Elapsed time during the whole program in minutes:", 
    #                                          (t1_stop-t1_start)/60)  


if __name__=='__main__':
    main()
