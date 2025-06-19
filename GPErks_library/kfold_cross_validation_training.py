import os 
import numpy as np
import torch
import json

# set logger and enforce reproducibility
from GPErks.log.logger import get_logger
from GPErks.utils.random import set_seed

from GPErks.gp.experiment import GPExperiment, load_experiment_from_config_file
from GPErks.train.emulator import GPEmulator

from GPErks.serialization.path import posix_path
from GPErks.train.snapshot import (
    EveryEpochSnapshottingCriterion,
    EveryNEpochsSnapshottingCriterion,
    NeverSaveSnapshottingCriterion
)

# define experiment
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from torchmetrics import MeanSquaredError, R2Score
from GPErks.utils.metrics import IndependentStandardError as ISE
from GPErks.gp.mean import LinearMean

# k-fold cross-validation training
from GPErks.perks.cross_validation import KFoldCrossValidation
from GPErks.train.early_stop import GLEarlyStoppingCriterion, NoEarlyStoppingCriterion

from GPErks.utils.array import tensorize
from GPErks.perks.inference import Inference

from sklearn.model_selection import train_test_split
from GPErks.gp.data.dataset import Dataset

FOLD = 5
SEED = 8
METRICS = [MeanSquaredError(), R2Score(), ISE()]
m_list = ["MSE", "R2Score", "ISE"]
WATCH_METRICS = [ISE(), R2Score()]



def kfold_cross_validation_training(loadpath,
                                    idx_feature,
                                    savepath,
                                    outpath,
                                    kernel="RBFKernel",
                                    parallel=True):


    """
    5-fold cross validation GPE training. 
    """


    if kernel not in ["RBFKernel","MaternKernel", "ScaleKernel"]:
    	raise Exception("Do not recognise kernel choice.")

    print("Kernel of choice: "+kernel)


    # ================================================================
    # Making the code reproducible
    # ================================================================
    
    log = get_logger()
    seed = SEED
    set_seed(SEED)
   

    # ================================================================
    # Loading the dataset
    # ================================================================

    X_ = np.loadtxt(loadpath + "X.txt", dtype=float)
    y_ = np.loadtxt(loadpath + "Y.txt", dtype=float)[:, int(idx_feature)]

    xlabels = np.loadtxt(loadpath + "xlabels.txt", dtype=str)
    ylabel = np.loadtxt(loadpath + "ylabels.txt", dtype=str)[int(idx_feature)]


    cmd = "mkdir -p " + savepath + "best_split/snapshot"
    os.system(cmd)

    config_file = savepath + "best_split/snapshot/emulator" + ".ini"
    
    snapshotpath = savepath + "best_split/snapshot/"
    
    cmd = "mkdir -p " + snapshotpath
    os.system(cmd)

    # split dataset in training and validation sets

    X, X_val, y, y_val = train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=seed
    )

    # build dataset
    dataset = Dataset(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
        x_labels=list(xlabels),
        y_label=ylabel,
        name="CanopyReflectance",
        descr="A reflectance model for the homogeneous plant canopy and its inversion (doi.org/10.1016/0034-4257(89)90015-1)"
    )

    # define experiment
    likelihood = GaussianLikelihood()
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
    metrics = METRICS

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,
        learn_noise=True
    )

    train_restart_template = "restart_{restart}"
    train_epoch_template = "epoch_{epoch}.pth"

    snapshot_file = train_epoch_template
    snpc = EveryEpochSnapshottingCriterion(
    posix_path(snapshotpath, train_restart_template),
    snapshot_file
    )



    # ================================================================
    # GPE training with K-fold cross-validation
    # ================================================================
    fold = FOLD

    device = "cpu"
    devices = [device]
    kfcv = KFoldCrossValidation(experiment, devices, n_splits=fold, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(
        max_epochs=1000, alpha=0.1, patience=8
    )
    best_model_dct, best_train_stats_dct = kfcv.train(
        optimizer,
        esc,

    )
    #print('**************:' ,kfcv.best_split)
    print(list(best_model_dct.keys())) 

    # resulting mean test scores
    kfcv.summary()

    data =  np.array([x for x in kfcv.best_test_scores_structured_dct.values()])
    for row in range(data.shape[0]):
    	np.savetxt(savepath + m_list[row] + "_cv.txt",data[row,:], fmt ='%1.4f', delimiter='\n')


    # check training stats at each split
    best_epochs = []
    for i, bts in best_train_stats_dct.items():
        # bts.plot(with_early_stopping_criterion=True)
        best_epochs.append( bts.best_epoch )

    # print( best_epochs )

    kfcv.emulator.experiment.metrics = WATCH_METRICS
    
    inference = Inference(kfcv.emulator)
    print('Inference Summary:')
    inference.summary()
    
    
     # check best-split emulator fitted hyperparameters
    kfcv.emulator.hyperparameters()

    # note the size differences between train and test sets here compared to original dataset train and val sets
    kfcv.emulator.experiment.dataset.summary()

    np.savetxt(savepath + "best_split/X_train.txt", kfcv.experiment.dataset.X_train[kfcv.best_split_idx[0]])
    np.savetxt(savepath + "best_split/y_train.txt", kfcv.experiment.dataset.y_train[kfcv.best_split_idx[0]])

    kfcv.emulator.experiment.save_to_config_file(config_file)

    best_model, best_train_stats,inference.scores_dct, _ = kfcv._train_split(
                                                        optimizer,
                                                        esc,
                                                        snpc,
                                                        kfcv.best_split,
                                                        device,
                                                        kfcv.experiment.dataset.X_train[kfcv.best_split_idx[0]],
                                                        kfcv.experiment.dataset.y_train[kfcv.best_split_idx[0]],
                                                        kfcv.experiment.dataset.X_train[kfcv.best_split_idx[1]],
                                                        kfcv.experiment.dataset.y_train[kfcv.best_split_idx[1]],
                                                        kfcv.best_split_idx[0],
                                                        kfcv.best_split_idx[1],
                                                    )
    
    # ================================================================
    # GPE training using the entire dataset
    # ================================================================

    cmd = "mkdir -p " + savepath + "snapshot"
    os.system(cmd)

    config_file = savepath + "snapshot/emulator" + ".ini"
    
    snapshotpath = savepath + "snapshot/"
    
    cmd = "mkdir -p " + snapshotpath
    os.system(cmd)

    X, X_test, y, y_test = train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=seed
    )

    # build a new dataset now including entire dataset
    del dataset
    dataset = Dataset(
        X,
        y,
        X_test=X_test,
    	y_test=y_test,
        x_labels=list(xlabels),
        y_label=ylabel
    )

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
    metrics = METRICS

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
    )

    train_restart_template = "restart_{restart}"
    train_epoch_template = "epoch_{epoch}.pth"

    snapshot_file = train_epoch_template
    snpc = EveryEpochSnapshottingCriterion(
    posix_path(snapshotpath, train_restart_template),
    snapshot_file
    )

    device = "cpu"
    emulator = GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    # making use of knowledge coming from the performed CV,
    # we run the training for an exact number of epochs
    max_epochs = int( np.mean(best_epochs) )  
    esc = NoEarlyStoppingCriterion(max_epochs)

    _, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc,
        snapshotting_criterion=snpc
    )

    #best_train_stats.plot(with_early_stopping_criterion=True)


    emulator.experiment.metrics = WATCH_METRICS

    print("**********************************")
    print("Emulator inference summary.......")
    inference = Inference(emulator)
    inference.summary()

    mets = np.array([x for x in inference.scores_dct.keys()])
    data =  np.array([x for x in inference.scores_dct.values()])

    out_data = dict(zip(mets,data))
  
    with open(savepath + "full_gpe_summary.txt" , 'w') as f: 
        json.dump(out_data, f)
    
    print("**********************************")
    print("Saving experiment to config file.......")

    # dump experiment in config file
    experiment.save_to_config_file(config_file)
    emulator.hyperparameters()

    # print("--------")
    # print(experiment.likelihood)
    # print(experiment.model)

    # np.savetxt(savepath + "likelihood.txt" ,experiment.likelihood)
    # np.savetxt(savepath + "model.txt" ,experiment.model)

    np.savetxt(savepath + "X_train.txt", X)
    np.savetxt(savepath + "y_train.txt", y)
