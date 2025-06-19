import os

import numpy as np
import diversipy as dp

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

SEED = 8

def history_matching(datapath,
                     matchpath,
                     gpepath,
                     cutoff,
                     waveno,
                     savepath,
                     previouswave="",
                     n_simuls=128,
                     n_tests=100000,
                     gpe_mode="full",
                     features_list_file=None,
                     X_test_file=None,
                     output_X=True,
                     maxno=1):

    """
    Performs history matching

    Args:
        - datapath: containig xlabels.txt and ylabels.txt
        - matchpath: containing exp_mean.txt and exp_std.txt to match
        - gpepath: containing the GPEs to use for each feature
        - cutoff: threshold for the implausibility measure to divide param space 
        - waveno: number of wave. Needed to save the files with correct tag
        - savepath: save the wave here
        - previouswave: to load up the test space
        - n_simuls: number of new simulation points to save (X_simul_waveno.txt)
        - n_tests: number of test points to save for next wave (X_test_waveno.txt)
        - gpe_mode: pick between best or full. Full is better because this way
                    you are sure you have a GPE that was trained on the full parameter range
        - simul_mode: how to select points to simulate for next wave (psa_select or random)
        - selection_target: psa_select option. Use random_uniform for better sampling
        - features_list_file: file containing the list of features to use. If set to None,
                              the code will look for features_idx_list_hm.txt in datapath
        - X_test_file: file containing the test points. If set to None, the file
                       is searched in the folder of the previous wave
        - output_X: True/False. If True, the code saves X_simul and X_test for the next wave
        - maxno: How many outputs to ignore in the history matching. If set to one,
                 the worst of all outputs for each point is considered and no output is 
                 disregarded. If set to 2, the second worst output is used for each point 
                 (in terms of implausibility measure), while the worst is ignored.

    """

    # ----------------------------------------------------------------
    # Make the code reproducible
    log = get_logger()
    seed = SEED
    set_seed(SEED)

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
                y_label=ylabels[idx]
                )
        
        #LOADING GPE 
        # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)

        experiment = load_experiment_from_config_file(
        config_file,
        dataset  # notice that we still need to provide the dataset used!
        )
        
        device = "cpu"
        emul = GPEmulator(experiment, device)

        best_model_file = os.readlink(snapshotpath + "best_model.pth")
        
        emul.load_state(best_model_file)

        # NOTICE: GPEs must have been trained using GPErks library (https://github.com/stelong/GPErks)
        emulator.append(emul)       

    I = get_minmax(
        X_train
    )  # get the spanning range for each of the parameters from the training dataset

    # ----------------------------------------------------------------

    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    if X_test_file is not None:
        print('Using provided X_test file. Not resampling.')
        X = np.loadtxt(X_test_file,dtype=float)
    else:
        if waveno == 1:
            X = lhd(
                I, n_tests
            )  # initial wave is performed on a big Latin hypercube design using same parameter ranges of the training dataset
        else:
            X = np.loadtxt(previouswave+"/X_test_"+str(waveno-1)+".txt", dtype=float)
            n_samples = X.shape[0]

    W.find_regions(
        X
    )  # enforce the implausibility criterion to detect regions of non-implausible and of implausible points
    # W.print_stats()  # show statistics about the two obtained spaces

    #--------------------------------------------------------------------

    # Added to enforce known inplausibility criterion EDP (S[:,5]) must be less than ESP (S[:,6])
    S = W.reconstruct_tests()
    n_samples = S.shape[0]
    print('orig: ',np.shape(W.NIMP))
    # print(np.shape(W.IMP))
    # print(np.shape(S))

    move_list =list(np.where(S[:,5] > S[:,6])[0])
    # print(np.shape(move_list))
    # print(np.shape(W.imp_idx))

    # Get the intersection
    moved_from_NIMP = np.intersect1d(W.nimp_idx, move_list)
    print(len(moved_from_NIMP))


    # Assuming W.imp_idx and move_list are NumPy arrays
    IMP_combined = np.concatenate((W.imp_idx, move_list))
    IMP_result = np.unique(IMP_combined)
    # print(np.shape(IMP_result))

    # print(IMP_result)

    NIMP_result = diff(range(n_samples), IMP_result)

    W.nimp_idx = NIMP_result
    W.NIMP = S[NIMP_result]
    W.imp_idx = IMP_result
    W.IMP = S[IMP_result]

    W.I[moved_from_NIMP] = 6.0

    W.print_stats()  # show statistics about the two obtained spaces


    #--------------------------------------------------------------------

    # examining GPE variance/exp. variance

    S = W.reconstruct_tests()
    n_samples = S.shape[0]
    output_dim = len(W.emulator)

    M = np.zeros((n_samples, output_dim), dtype=float)
    V = np.zeros((n_samples, output_dim), dtype=float)
    PVn_out = np.zeros((n_samples, output_dim), dtype=float)
    for j, emul in enumerate(W.emulator):
        mean, std = emul.predict(S)
        var = np.power(std, 2)
        M[:, j] = mean
        V[:, j] = var
        I = np.zeros((n_samples,), dtype=float)
        PV = np.zeros((n_samples,), dtype=float)
        for i in range(n_samples):
            In = np.sqrt(
                (np.power(M[i, :] - W.mean, 2)) / (V[i, :] + W.var)
            )
            PVn = V[i, :] / W.var
            PVn_out[i,:] = PVn

    np.savetxt(savepath+"/PVn_"+str(waveno)+".txt", PVn_out, fmt='%g')

    # ----------------------------------------------------------------
    # To continue on the next wave:

    # (0) Save an exact copy of the wave before manipulating its structure
    W0 = W.copy()
    #W0.print_stats()
    W0.save(
        savepath + "/wave_"+str(waveno)
    )  # this is a good moment to save the wave object if you need it later for other purposes (see Appendix)
    #
    # (1) Select points to be simulated and points to be used as tests for the next wave from the current non-implausible region

    if output_X:

        n_avail_nimps = len(W0.nimp_idx)  # we currently have only this number of NIMP points
        if n_tests + n_simuls > n_avail_nimps:  # if they are not enough
            n_total_points = n_tests + n_simuls
            W.augment_nimp(n_total_points)  # use the 'cloud technique' to generate new NIMP points starting from the existing one
         
        print('here:',len(W.nimp_idx))
        X_simul, X_test = W.get_nimps(n_simuls)  # now we can get the requested datasets and save them
        np.savetxt(savepath+"/X_simul_"+str(waveno)+".txt", X_simul, fmt='%g')
        np.savetxt(savepath+"/X_test_"+str(waveno)+".txt", X_test, fmt='%g')   

       
        # (2) Simulate the selected points
        # (3) Add the simulated points and respective results to the training dataset used in the previous wave
        # (3) Train GPEs on the new, augmented training dataset
        # (4) Start a new wave of history matching, where the initial parameter space to be split into non-implausible and implausible regions 
        #     is no more a Latin hypercube design but is now the non-implausible region obtained in the previous wave and saved as:
        # n_tests = 100000  # number of test points we want for the next wave (from the current non-implausible region)
        
        
        # ----------------------------------------------------------------
        # Appendix A - Wave object loading
        # # You can load a wave object by providing the same data used to instantiate the wave: emulator, Itrain, cutoff, maxno, mean, var. This is normally done when you need to re-run the wave differently.
        # # Alternatively, you can load the wave object by providing no data at all, just to better examine its internal structure:
        # W = hm.Wave()
        # W.load(f"./wave_{waveno}")
        # W.print_stats()

        # # This is the list of the loaded wave object attributes:
        # print(W.__dict__.keys())

        # # Noteworthy attributes are:
        # # W.I = implausibility measure obtained for each point in the test set
        # # W.PV = percentage emulator variance over experimental variance at each point (given as a fraction)
        # # W.NIMP = non-implausible region
        # # W.nimp_idx = indices of the initial test set which resulted to be non-implausible
        # # W.IMP = implausible region
        # # W.imp_idx = indices of the initial test set which resulted to be implausible
        # # W.simul_idx = indices of W.NIMP that were selected to be simulated for the next wave
        # # W.nsimul_idx = indices of W.NIMP which were not selected for simulations (the respective points will appear in the test set of the next wave instead)

        # # The original test set is not stored as an attribute to save space. However, this information can still be retrieved from stored attributes as:
        # # X_test = W.reconstruct_tests()

        # # ----------------------------------------------------------------
        # # Appendix B - Checking generated test dataset for next wave
        # X_nimp = W.NIMP
        # X_test = np.loadtxt(f"./X_test_{waveno}.txt", dtype=float)
        # X_simul = np.loadtxt(f"./X_simul_{waveno}.txt", dtype=float)

        # param = [4, 6]  # plot only 2 dimensions of the entire parameter space
        # subset_idx = list(np.random.randint(0, X_test.shape[0], size=X_simul.shape[0]))  # plot only a subset of the entire X_test

        # colors = interp_col(get_col("blue"), 4)

        # fig, axis = plt.subplots(1, 1)
        # axis.scatter(X_nimp[:, param[0]], X_nimp[:, param[1]], fc=colors[1], ec=colors[1], label=f"X_nimp of wave {waveno}")
        # axis.scatter(X_test[subset_idx, param[0]], X_test[subset_idx, param[1]], fc=colors[-1], ec=colors[-1], label=f"X_test for wave {waveno+1}")
        # axis.scatter(X_simul[:, param[0]], X_simul[:, param[1]], fc='r', ec='r', label=f"X_simul for wave {waveno+1}")
        # axis.set_xlabel(xlabels[param[0]], fontsize=12)
        # axis.set_ylabel(xlabels[param[1]], fontsize=12)
        # axis.legend()
        # fig.tight_layout()
        # plt.show()  # test points + simul points for NEXT wave are all within nimp space of the CURRENT wave


def intersect_waves(wave_1,
                    wave_2,
                    n_tests=100000,
                    scale=0.1):

    # wave_1 & wave_2: wave objects with emulators loaded

    # start with the NIMP of the first wave
    NROY = np.copy(wave_1.NIMP)
    lbounds = wave_1.Itrain[:, 0]
    ubounds = wave_1.Itrain[:, 1]

    initial_NIMP_size = NROY.shape[0]

    # this step checks that that the NIMP of 
    # the first wave is also NIMP for the second wave 
    wave_2.find_regions(NROY)
    NROY = wave_2.NIMP

    corrected_NIMP_size = NROY.shape[0]

    print('Initial rejection reduced NIMP from '+str(initial_NIMP_size)+' to '+str(corrected_NIMP_size))


    print(
        f"\nRequested points: {n_tests}\nAvailable points: {NROY.shape[0]}\nStart searching..."
    )

    count = 0
    a, b = (
        NROY.shape[0] if NROY.shape[0] < n_tests else n_tests,
        n_tests - NROY.shape[0] if n_tests - NROY.shape[0] > 0 else 0,
    )
    print(
        f"\n[Iteration: {count:<2}] Found: {a:<{len(str(n_tests))}} ({'{:.2f}'.format(100*a/n_tests):>6}%) | Missing: {b:<{len(str(n_tests))}}"
    )

    while NROY.shape[0] < n_tests:
        count += 1

        I = get_minmax(NROY)
        SCALE = scale * np.array(
            [I[i, 1] - I[i, 0] for i in range(NROY.shape[1])]
        )

        temp = np.random.normal(loc=NROY, scale=SCALE)
        while True:
            l = []
            for i in range(temp.shape[0]):
                d1 = temp[i, :] - lbounds
                d2 = ubounds - temp[i, :]
                if (
                    np.sum(np.sign(d1)) != temp.shape[1]
                    or np.sum(np.sign(d2)) != temp.shape[1]
                ):
                    l.append(i)
            if l:
                temp[l, :] = np.random.normal(loc=NROY[l, :], scale=SCALE)
            else:
                break

        wave_1.find_regions(temp)
        wave_2.find_regions(wave_1.NIMP)
        NROY = np.vstack((NROY, wave_2.NIMP))

        a, b = (
            NROY.shape[0] if NROY.shape[0] < n_tests else n_tests,
            n_tests - NROY.shape[0] if n_tests - NROY.shape[0] > 0 else 0,
        )
        print(
            f"[Iteration: {count:<2}] Found: {a:<{len(str(n_tests))}} ({'{:.2f}'.format(100*a/n_tests):>6}%) | Missing: {b:<{len(str(n_tests))}}"
        )

    print("\nDone.")
    TESTS = np.vstack(
        (
            NROY[: corrected_NIMP_size],
            dp.subset.psa_select(NROY[corrected_NIMP_size :], n_tests - corrected_NIMP_size, selection_target='random_uniform'),
        )
    )
    return TESTS

def compute_impl_noGPE(Y,
                       exp_mean,
                       exp_var,
                       maxno):

    """
    Computes implausibility measure without GPEs as

    I(x) = sqrt((Y(x)-exp_mean)^2/exp_var)

    Args:
        - Y: output simulated values
        - exp_mean: vector of experimental means
        - exp_var: vector of experimental variance
        - maxno: if 1, it takes the worst implausibility over the outputs
                 if 2, it takes the second worst implausibility over the outputs and so on

    Outputs:
        - I: vector with the implausibility measure computed for each sample
 
    """

    I = np.zeros((Y.shape[0],), dtype=float)
    for i in range(Y.shape[0]):
        In = np.sqrt(
            (np.power(Y[i, :] - exp_mean, 2)) / (exp_var)
        )

        I[i] = np.sort(In)[-maxno]

    PV = np.zeros((Y.shape[0],), dtype=float)

    return I,PV

def find_regions_noGPE(X,I,cutoff):

    """
    Computes NIMP and IMP
    Args:
        - X: samples
        - I: implausibility measure for each sample
        - cutoff: threshold on implausibility measure

    Outputs:
        - NIMP: non-implausible samples
        - IMP: implausible samples
 
    """
    
    l = np.where(I < cutoff)[0]
    nl = diff(range(I.shape[0]), l)

    NIMP = X[l]
    IMP = X[nl]

    return NIMP,IMP,l,nl

def history_matching_noGPE(datapath,
                           matchpath,
                           cutoff,
                           waveno,
                           savepath,
                           previouswave="",
                           n_simuls=128,
                           simul_mode="psa_select",
                           selection_target="random_uniform",
                           features_list_file=None,
                           X_test_file=None,
                           Y_test_file=None):

    """
    Performs history matching without GPEs, using only simulations.
    This means that the I(x) comes only from the data.

    Args:
        - datapath: containig xlabels.txt and ylabels.txt
        - matchpath: containing exp_mean.txt and exp_std.txt to match
        - cutoff: threshold for the implausibility measure to divide param space 
        - waveno: number of wave. Needed to save the files with correct tag
        - savepath: save the wave here
        - previouswave: to load up the test space 
        - n_simuls: number of new simulation points to save (X_simul_waveno.txt)
        - n_tests: number of test points to save for next wave (X_test_waveno.txt)
        - simul_mode: how to select points to simulate for next wave (psa_select or random)
        - selection_target: psa_select option. Use random_uniform for better sampling
        - features_list_file: file containing the list of features to use. If set to None,
                              the code will look for features_idx_list_hm.txt in datapath
        - X_test_file: file containing the test points. If set to None, the file
                       is searched in the folder of the previous wave
        - Y_test_file: file containing the simulated output for all test points. If set to None, 
                       the file is searched in the folder of the previous wave
 
    """
    # ----------------------------------------------------------------
    # Make the code reproducible
    log = get_logger()
    seed = SEED
    set_seed(SEED)

    # ----------------------------------------------------------------
    # Load experimental values (mean +- std) you aim to match
    if not os.path.exists(matchpath + "exp_mean.txt") or not os.path.exists(matchpath + "exp_std.txt"):
        raise Exception('matchpath should contain exp_mean.txt and exp_std.txt.')
    exp_mean = np.loadtxt(matchpath + "exp_mean.txt", dtype=float)
    exp_std = np.loadtxt(matchpath + "exp_std.txt", dtype=float)
    exp_var = np.power(exp_std, 2)

    # ----------------------------------------------------------------
    # Load input parameters and output features' names
    if not os.path.exists(datapath + "ylabels.txt"):
        raise Exception('datapath should contain ylabels.txt.')
    ylabels = read_labels(datapath + "ylabels.txt")
    features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}

    # ----------------------------------------------------------------
    if features_list_file is None:
        features_list_file = datapath+"features_idx_list_hm.txt"
    active_idx = np.loadtxt(features_list_file,dtype=int)

    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]
    ylabels = [ylabels[idx] for idx in active_idx]

    print('History matching using features ')
    for ylab in ylabels:
        print(ylab)

    # ----------------------------------------------------------------
    print('Reading simulated test points...')

    if X_test_file is None:
        X_test_file = previouswave+"/X_test_"+str(waveno-1)+".txt"

    if Y_test_file is None:
        Y_test_file = previouswave+"/Y_test_"+str(waveno-1)+".txt"

    X_test = np.loadtxt(X_test_file, dtype=float)
    Y_test = np.loadtxt(Y_test_file, dtype=float, usecols=active_idx)

    if X_test.shape[0]!=Y_test.shape[0]:
        raise Exception('Careful! X and Y test have different number of samples - some of your simulations crashed!')

    print('Computing implausibility measure...')
    maxno = 1 
    I,PV = compute_impl_noGPE(Y_test,
                              exp_mean,
                              exp_var,
                              maxno)
    NIMP,IMP,nimp_idx,imp_idx = find_regions_noGPE(X_test,I,cutoff)
    Itrain = get_minmax(X_test) 

    # ----------------------------------------------------------------

    W = hm.Wave(
        cutoff=cutoff,
        Itrain=Itrain,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var
    ) 
    W.I = I
    W.PV = PV
    W.NIMP = NIMP
    W.IMP = IMP
    W.nimp_idx = nimp_idx
    W.imp_idx = imp_idx
    W.n_samples = X_test.shape[0]
    W.input_dim = X_test.shape[1]
    W.print_stats() 

    if n_simuls>0:
        SIMULS = W.get_nimps(n_simuls)  # actual matrix of selected points
        np.savetxt(savepath+"/X_simul_"+str(waveno)+".txt", SIMULS, fmt='%g')

    W.save(savepath+"/wave_"+str(waveno))

