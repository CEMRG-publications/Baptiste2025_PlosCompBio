import numpy as np
import pandas as pd

def gaussian_ll(
    y,
    mean,
    variance
    ):
    ll = -0.5*((y-mean)**2)/variance - 0.5*np.log(variance) - 0.5*np.log(2.*np.pi)
    return ll


def ensemble_log_likelihood_obs_error(
    candidateInput, 
    emulators, 
    exp_mean, 
    exp_vars
    )->np.ndarray:
    nMod = len(emulators)
    nP = candidateInput.shape[0]
    likelihood_eval = np.zeros((nMod,nP))
    inputNorm = candidateInput.reshape((1,-1))
    
    for i in range(nMod):
        # emulators[i].experiment.likelihood.eval()
        # emulators[i].experiment.model.eval()
        exp_var = exp_vars[i]
        
        m, k = emulators[i].predict(inputNorm)
        k = k**2
           
        #########
        emulator_mean     = m
        total_variance = k + exp_var
        #########
        
    
        likelihood_manual=gaussian_ll(exp_mean[i],emulator_mean,total_variance)        
        likelihood_eval[i,:] = likelihood_manual
        
        
    return likelihood_eval

def log_likelihood(
    theta, 
    emulators,
    exp_mean,
    exp_vars
    )-> float:

	ll = np.sum((ensemble_log_likelihood_obs_error(theta, emulators, exp_mean,exp_vars)))
	return ll

def log_prior(
    theta, 
    boundsMaxMin
    )-> float:
    
    if (np.array(boundsMaxMin)[:,0]<theta).all() and (theta<np.array(boundsMaxMin)[:,1]).all():
        return 0.0
    return -np.inf

def log_prob(
    theta, 
    emulators, 
    exp_mean, 
    exp_vars, 
    boundsMaxMin
    )-> float:
    
    lp = log_prior(theta, boundsMaxMin)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, emulators, exp_mean, exp_vars)




