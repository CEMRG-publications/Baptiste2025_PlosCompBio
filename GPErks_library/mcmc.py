import os
import emcee

import numpy as np
import diversipy as dp
import pandas as pd

from GPErks.log.logger import get_logger
from GPErks.utils.random import set_seed

import torch
from GPErks.serialization.path import posix_path
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import load_experiment_from_config_file
from GPErks.train.emulator import GPEmulator

from Historia.history import hm
from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.shared.indices_utils import diff, whereq_whernot

import scipy
from GPErks_library import GPE_ensemble as GPE

SEED = 8

def mcmc(datapath,
         matchpath,
         gpepath,
         gpe_mode="full",
         features_list_file=None):
    
    # ----------------------------------------------------------------
    # Load experimental values (mean +- std) you aim to match
    exp_mean = np.loadtxt(matchpath + "exp_mean.txt", dtype=float)
    exp_std = np.loadtxt(matchpath + "exp_std.txt", dtype=float)
    exp_var = np.power(exp_std, 2)
    
    # ----------------------------------------------------------------
    # Load input parameters and output features' names
    xlabels = read_labels(datapath + "xlabels.txt")
    ylabels = read_labels(datapath + "ylabels.txt")
    features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}

    # ----------------------------------------------------------------
    if features_list_file is None:
        features_list_file = datapath+"features_idx_list_hm.txt"

    active_idx = np.loadtxt(features_list_file,dtype=int)

    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]

    if active_idx.size>1:
        ylabels = [ylabels[idx] for idx in active_idx]
    else:
        ylabels = [ylabels[active_idx]]

    print('History matching using features ')
    for ylab in ylabels:
        print(ylab)

    # Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature to match
    emulator = []

    if active_idx.size>1:
        
        y_train_full = np.loadtxt(gpepath + '0/y_train.txt', dtype=float)
        y_train_full = np.reshape(y_train_full,(len(y_train_full), 1))

        for idx in active_idx:
            if gpe_mode=="full":
                loadpath = gpepath + str(idx) + "/"
                X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
                y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
                y_train_full =  np.concatenate((y_train_full, y_train.reshape(len(y_train),1)), axis=1)

            elif gpe_mode=="best":
                loadpath = gpepath +  str(idx) + "/best_split/"
                X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
                y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
           
            
            y_train_full =  np.delete(y_train_full,0 ,1)  
            X_train = pd.DataFrame(X_train[:,:5])
            y_train_full = pd.DataFrame(y_train_full)
            print(X_train)

            # #LOADING GPE 
            # # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)
            # snapshotpath = loadpath + "snapshot/"
            # config_file = loadpath + "snapshot/emulator" + ".ini"
            
            # dataset = Dataset(
            #         X_train,
            #         y_train,
            #         x_labels=xlabels,
            #         y_label=ylabels[idx]
            #         )
                
            # experiment = load_experiment_from_config_file(
            # config_file,
            # dataset  # notice that we still need to provide the dataset used!
            # )
            
            # device = "cpu"
            # emul = GPEmulator(experiment, device)

            # best_model_file = os.readlink(snapshotpath + "best_model.pth")
            
            # emul.load_state(best_model_file)

            # emulator.append(emul)       

    else:
        if gpe_mode=="full":
            loadpath = gpepath + str(active_idx) + "/"
            X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
            y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
        
        elif gpe_mode=="best":
            loadpath = gpepath +  str(active_idx) + "/best_split/"
            X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
            y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
     

        # snapshotpath = loadpath + "snapshot/"
        # config_file = loadpath + "snapshot/emulator" + ".ini"
        
        # dataset = Dataset(
        #         X_train,
        #         y_train,
        #         x_labels=xlabels,
        #         y_label=ylabels[idx]
        #         )
        
        # #LOADING GPE 
        # # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)

        # experiment = load_experiment_from_config_file(
        # config_file,
        # dataset  # notice that we still need to provide the dataset used!
        # )
        
        # device = "cpu"
        # emul = GPEmulator(experiment, device)

        # best_model_file = os.readlink(snapshotpath + "best_model.pth")
        
        # emul.load_state(best_model_file)

        # # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)
        # emulator.append(emul)  

    # ------------------------------------------------------------------------]
    # generate ensemble


    emulator_ensemble = GPE.ensemble(X_train, y_train_full,mean_func="linear",training_iter=500)

    #--------------------------------------------------------------------------
    # create prior distribution using input parameters

    regions = ["anterior", "posterior", "septum", "lateral", "roof"]
    # for i, region in enumerate(regions):
    anterior = X_train[0]
    posterior = X_train[1]
    septum = X_train[2]
    lateral = X_train[3]
    roof = X_train[4]

    # anterior_prior = scipy.stats.norm.pdf(anterior,np.mean(anterior),1)
    # posterior_prior = scipy.stats.norm.pdf(posterior,np.mean(posterior),1)
    # septum_prior = scipy.stats.norm.pdf(septum,np.mean(septum),1)
    # lateral_prior = scipy.stats.norm.pdf(lateral,np.mean(lateral),1)
    # roof_prior = scipy.stats.norm.pdf(roof,np.mean(roof),1)

    # Nx = X_train.shape[0]

    # joint_prior = scipy.stats.multivariate_normal.pdf(np.stack((anterior,posterior,septum,lateral,roof), axis=-1),[np.mean(anterior),np.mean(posterior),np.mean(septum),np.mean(lateral),np.mean(roof)],1*np.identity(5))
    # print(joint_prior.shape)

    #--------------------------------------------------------------------------
    #create posterior distribution using input parameters

    # danterior = (np.max(anterior)-np.min(anterior))/Nx
    # dposterior = (np.max(posterior)-np.min(posterior))/Nx
    # dseptum = (np.max(septum)-np.min(septum))/Nx
    # dlateral = (np.max(lateral)-np.min(lateral))/Nx
    # droof = (np.max(roof)-np.min(roof))/Nx

    # anterior_posterior = likelihood.reshape(Nx,Nx,Nx)*anterior_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*anterior_prior)*(danterior*dposterior*dseptum*dlateral*droof))
    # posterior_posterior = likelihood.reshape(Nx,Nx,Nx)*posterior_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*posterior_prior)*(danterior*dposterior*dseptum*dlateral*droof))
    # septum_posterior = likelihood.reshape(Nx,Nx,Nx)*septum_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*septum_prior)*(danterior*dposterior*dseptum*dlateral*droof))
    # lateral_posterior = likelihood.reshape(Nx,Nx,Nx)*lateral_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*lateral_prior)*(danterior*dposterior*dseptum*dlateral*droof))
    # roof_posterior = likelihood.reshape(Nx,Nx,Nx)*roof_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*roof_prior)*(danterior*dposterior*dseptum*dlateral*droof))

    # joint_posterior =likelihood.reshape(Nx,Nx,Nx)*joint_prior / (np.sum(likelihood.reshape(Nx,Nx,Nx)*joint_prior)*(danterior*dposterior*dseptum*dlateral*droof))
    

    # ----------------------------------------------------------------------------

    ndim = 5
    nwalkers = 10
    p0 = np.random.multivariate_normal((np.mean(anterior),np.mean(posterior),np.mean(septum),np.mean(lateral),np.mean(roof)), 0.01*np.identity(5), size=(nwalkers))
    y_val = y_train_full
    print(p0)


    def log_prob(x, mu, cov):
        diff = x - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    
    means = np.array([np.mean(anterior),np.mean(posterior),np.mean(septum),np.mean(lateral),np.mean(roof)])
    
   






