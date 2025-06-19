import os
import sys
import seaborn as sns
import skopt
import tqdm

import numpy as np
from scipy.special import binom
from scipy.stats import norm
import networkx as nx

import random
from itertools import combinations

import torch
from SALib.analyze import sobol
from SALib.sample import saltelli

from GPErks.log.logger import get_logger
from GPErks.utils.random import set_seed

# from gpytGPE.gpe import GPEmul
# from gpytGPE.utils.design import get_minmax, read_labels
# from gpytGPE.utils.plotting import gsa_box, gsa_donut, gsa_network
# from gpytGPE.utils.design import lhd

from Historia.history import hm
# from GSA_library.plotting import plot_dataset
# from GSA_library import sobol_analyze_NIMP
# from GSA_library.saltelli_pick_sampling import sample_NIMP

from GPErks.utils.plotting import get_col, interp_col
from GPErks.utils.array import get_minmax
from GPErks.constants import (
    DEFAULT_GSA_CONF_LEVEL,
    DEFAULT_GSA_N,
    DEFAULT_GSA_N_BOOTSTRAP,
    DEFAULT_GSA_N_DRAWS,
    DEFAULT_GSA_SKIP_VALUES,
    DEFAULT_GSA_THRESHOLD,
    DEFAULT_GSA_Z,
)
from GPErks.serialization.path import posix_path
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import load_experiment_from_config_file
from GPErks.train.emulator import GPEmulator



SEED = 8
EMUL_TYPE = "full"  # possible choices are: "full", "best"


def global_sobol_sensitivity_analysis(loadpath,
                                      idx_feature,
                                      savepath,
                                      sample_method='sobol',
                                      calc_second_order=True,
                                      uncertainty=True,
                                      n_factor=1):


    # ================================================================
    # Making the code reproducible
    # ================================================================
    
    log = get_logger()
    seed = SEED
    set_seed(SEED)
   
    # ================================================================
    # GPE loading
    # ================================================================
    
    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",savepath]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find '+f)
        else:
            print(f+' found.')

    xlabels = np.loadtxt(loadpath + "xlabels.txt", dtype=str)
    ylabel = np.loadtxt(loadpath + "ylabels.txt", dtype=str)[int(idx_feature)]

    emul_type = EMUL_TYPE

    print('Using '+emul_type+' GPE for sensitivity analysis...')

    if emul_type == "best":
        savepath = savepath + "best_split/"

    if not os.path.exists(savepath + "/X_train.txt"):
        raise Exception('Cannot find '+savepath+'/X_train.txt')
    if not os.path.exists(savepath + "/y_train.txt"):
        raise Exception('Cannot find '+savepath+'/y_train.txt') 
               
    X_train = np.loadtxt(savepath + "/X_train.txt", dtype=float)
    y_train = np.loadtxt(savepath + "/y_train.txt", dtype=float)

    snapshotpath = savepath + "snapshot/"
    config_file = savepath + "snapshot/emulator" + ".ini"
    
    dataset = Dataset(
            X_train,
            y_train,
            x_labels=list(xlabels),
            y_label=ylabel
            )
        
    experiment = load_experiment_from_config_file(
    config_file,
    dataset  # notice that we still need to provide the dataset used!
    )
    
    device = "cpu"
    emulator = GPEmulator(experiment, device)
    emulator.hyperparameters()
    
    best_model_file = os.readlink(snapshotpath + "best_model.pth")
    
    emulator.load_state(best_model_file)
    emulator.hyperparameters()

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================
    
    d = dataset.input_size
    index_i= dataset.x_labels
    index_ij = [list(c) for c in combinations(index_i, 2)]
    ylabel= dataset.y_label
    minmax = get_minmax(dataset.X_train)
    n = DEFAULT_GSA_N
    n_draws = DEFAULT_GSA_N_DRAWS


    problem = {
        "num_vars": d,
        "names": index_i,
        "bounds": minmax,
    }

    if sample_method=='sobol':
        X = saltelli.sample(problem, n, calc_second_order=True,skip_values=DEFAULT_GSA_SKIP_VALUES)


    if uncertainty:

        Y = emulator.sample(X, n_draws)

        ST= np.zeros((0, d), dtype=float)
        S1= np.zeros((0, d), dtype=float)
        S2= np.zeros((0, int(binom(d, 2))), dtype=float)

        ST_std = np.zeros((0, d), dtype=float)
        S1_std = np.zeros((0, d), dtype=float)
        S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)

        for i in tqdm.tqdm(range(n_draws)):
            S = sobol.analyze(
            problem,
            Y[i],
            calc_second_order=True,
            num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
            conf_level=DEFAULT_GSA_CONF_LEVEL,
            parallel=False,
            n_processors=None,
            seed=seed)

            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            ST = np.vstack((ST, T_Si["ST"].reshape(1,-1)))
            S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1)))
            if calc_second_order:
                S2 = np.vstack((S2, np.array(second_Si["S2"]).reshape(1, -1)))

            ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1)                / DEFAULT_GSA_Z))
            S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1)            / DEFAULT_GSA_Z))
            if calc_second_order:
                S2_std = np.vstack((S2_std, (np.array(second_Si["S2_conf"]).reshape(1, -1)/ DEFAULT_GSA_Z)))


        np.savetxt(savepath + "/STi.txt", ST, fmt="%.6f")
        np.savetxt(savepath + "/Si.txt", S1, fmt="%.6f")
        np.savetxt(savepath + "/Sij.txt", S2, fmt="%.6f")

        np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
        np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
        np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")

    else:

        Y, std = emul.predict(X)

        S = sobol.analyze(
            problem,
            Y,
            calc_second_order=True,
            num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
            conf_level=DEFAULT_GSA_CONF_LEVEL,
            parallel=False,
            n_processors=None,
            seed=seed)

        T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S) 

        ST = T_Si["ST"].reshape(1, -1)
        S1 = first_Si["S1"].reshape(1, -1) 

        if calc_second_order:
            S2 = np.array(second_Si["S2"]).reshape(1, -1)
        else:
            S2 = np.zeros((0, int(binom(d, 2))), dtype=float)

        ST_std = T_Si["ST_conf"].reshape(1, -1) / z
        S1_std = first_Si["S1_conf"].reshape(1, -1) / z    

        if calc_second_order:
            S2_std = np.array(second_Si["S2_conf"]).reshape(1, -1) / z
        else:
            S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)
           
        np.savetxt(savepath + "STi.txt", ST, fmt="%.6f")
        np.savetxt(savepath + "Si.txt", S1, fmt="%.6f")
        np.savetxt(savepath + "Sij.txt", S2, fmt="%.6f")    

        np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
        np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
        np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")




