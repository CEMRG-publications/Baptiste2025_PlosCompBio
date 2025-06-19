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
from GPErks.serialization.labels import read_labels_from_file


def normalise_data(theta, output_val, emulator):
    
    
    x_data = emulator[0].experiment.dataset.X_train

    norm_arr = []
    
    for col in range(0,x_data.shape[1]):
    
        diff = max(x_data[:,col]) - min(x_data[:,col])

        norm_input_value = (theta[col] - min(x_data[:,col]))/diff
        norm_arr.append(norm_input_value)
    
    norm_arr = np.reshape(norm_arr, (1,len(norm_arr)))
    x_df = pd.DataFrame(data = norm_arr)
    
    ### -------------------------------------------------------------------
    
    
    norm_arr = []
    
    for j in range(len(emulator)):
        
        y_data = emulator[j].experiment.dataset.y_train

        norm_output_value = (output_val[j]-np.mean(y_data))/(np.std(y_data))
        norm_arr.append(norm_output_value)
        
    norm_arr = np.reshape(norm_arr, (1,len(norm_arr)))
    y_df = pd.DataFrame(data = norm_arr)
   
    
    return x_df, y_df


def ensemble_log_likelihood(theta, emulator, output_val):
    n_features = len(emulator)
    likelihood_eval = np.zeros(n_features)
    inputNorm, outputNorm = normalise_data(theta, output_val, emulator)
    # emulator[0].experiment.likelihood.eval()
    # emulator[0].experiment.model.eval()
    # print(emulator[0].experiment.model(torch.tensor(inputNorm.values).float()))
    for i in range(n_features):
        emulator[i].experiment.likelihood.eval()
        emulator[i].experiment.model.eval()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(emulator[i].experiment.likelihood, emulator[i].experiment.model)
        likelihood_eval[i] = (mll(emulator[i].experiment.model(torch.tensor(inputNorm.values).float()),torch.tensor(outputNorm.iloc[-1,i]).float()).detach().numpy())
        # print(emulator[i].experiment.model(torch.tensor(inputNorm.values).float()))
    
    return likelihood_eval 


def ensemble_log_likelihood_obs_error_old(candidateInput,emulator, outputVal,sigma2):
    
   
    nMod = len(emulator)
    nDim = emulator[0].experiment.dataset.y_train.shape[0]
    nP = candidateInput.shape[0]
    # print('***')
    # print(nMod,nDim,nP)

    likelihood_eval = torch.zeros((nMod,nP))
    inputNorm,outputNorm = normalise_data(candidateInput,outputVal, emulator, xlabels, ylabels)
    inputNorm = torch.tensor(inputNorm.values).float()
    # outputNorm = torch.tensor(outputNorm.values)
    inputNorm=inputNorm.float()
    # outputNorm=outputNorm.float()
    
    
    for i in range(nMod):
        emulator[i].experiment.likelihood.eval()
        emulator[i].experiment.model.eval()
        sigma = sigma2[i]
        m = emulator[i].experiment.likelihood(emulator[i].experiment.model(inputNorm)).mean
        k = emulator[i].experiment.likelihood(emulator[i].experiment.model(inputNorm)).covariance_matrix.diag()

        y_data = emulator[i].experiment.dataset.y_train
        y_data_mean = np.mean(y_data)
        y_data_std = np.std(y_data)
        
        likelihood_manual=-0.5*((outputVal[i]-(y_data_std*m+y_data_mean))**2)/(y_data_std*k+sigma) -0.5*nDim*torch.log(y_data_std*k+sigma) #- 0.5*nDim*torch.log(torch.tensor(2*torch.pi))
        likelihood_eval[i,:] = likelihood_manual
        
    return likelihood_eval


def gaussian_ll(y,mean,variance):

    ll = -0.5*((y-mean)**2)/variance - 0.5*torch.log(variance) - 0.5*torch.log(torch.tensor(2*torch.pi))
    return ll


def ensemble_log_likelihood_obs_error(candidateInput,emulator, outputVal,sigma2):
    
    nMod = len(emulator)
    nDim = emulator[0].experiment.dataset.y_train.shape[0]
    nP = candidateInput.shape[0]
    likelihood_eval = torch.zeros((nMod,nP))
    inputNorm,outputNorm = normalise_data(candidateInput,outputVal, emulator)
    inputNorm = torch.tensor(inputNorm.values).float()
    inputNorm=inputNorm.float()
    
    
    for i in range(nMod):
        emulator[i].experiment.likelihood.eval()
        emulator[i].experiment.model.eval()
        sigma = sigma2[i]
        m = emulator[i].experiment.likelihood(emulator[i].experiment.model(inputNorm)).mean
        k = emulator[i].experiment.likelihood(emulator[i].experiment.model(inputNorm)).covariance_matrix.diag()
   
        y_data = emulator[i].experiment.dataset.y_train
        y_data_mean = np.mean(y_data)
        y_data_std = np.std(y_data)

        mean = y_data_std*m+y_data_mean
        variance = (y_data_std**2)*k+sigma
    
        likelihood_manual=gaussian_ll(outputVal[i],mean,variance)
        
        #likelihood_manual=-0.5*((outputVal[:,i]-(self.training_output_STD[i]*m+self.training_output_mean[i]))**2)/((self.training_output_STD[i]**2)*k+sigma) -0.5*torch.log((self.training_output_STD[i]**2)*k+sigma) - 0.5*torch.log(torch.tensor(2*torch.pi))
        likelihood_eval[i,:] = likelihood_manual
        
        
    return likelihood_eval

def log_likelihood(x,emulator,y_val,sigma2):

	ll = np.sum((ensemble_log_likelihood_obs_error(x, emulator, y_val,sigma2)).detach().numpy())
	return ll

def log_prior(theta, boundsMaxMin):
    
    if (np.array(boundsMaxMin)[:,0]<theta).all() and (theta<np.array(boundsMaxMin)[:,1]).all():
    # if (np.array(boundsMaxMin)[:,0]*0.5<theta).all() and (theta<np.array(boundsMaxMin)[:,1]*4).all():
        return 0.0
    return -np.inf

def log_prob(theta, emulator,y_val,sigma2, boundsMaxMin):
    #print(theta)
    lp = log_prior(theta, boundsMaxMin)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta,emulator, y_val, sigma2)




