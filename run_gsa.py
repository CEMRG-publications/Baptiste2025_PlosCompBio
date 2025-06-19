import os

import numpy as np

from GPErks.serialization.labels import read_labels_from_file

from GPErks_library import kfold_cross_validation_training
from GPErks_library import global_sobol_sensitivity_analysis
from GPErks_library import gsa_parameters_ranking

from GPErks_library.plotting import *
from GPErks_library.gsa_plotting import * 

def main():

	basefolder = './p0001_ani/'

	waveno = 1

	idx_feature = list(np.loadtxt(basefolder+"/data/wave1/features_idx_list.txt",dtype=int))

	# ================================================================
    # GPE TRAINING
    # ================================================================
	for idx in idx_feature:

		loadpath = basefolder + 'data/wave' + str(waveno) + '/'
		outpath = basefolder + 'output/wave' + str(waveno) + '/'
		savepath = basefolder + 'output/wave' + str(waveno) + '/' + str(idx) + '/'

		if not os.path.exists(savepath+"snapshot"):
			cmd = 'mkdir -p ' + savepath
			os.system(cmd)

			kfold_cross_validation_training.kfold_cross_validation_training(loadpath,idx,savepath,outpath)

    # ================================================================
    # GSA
    # ================================================================
	loadpath = basefolder + 'data/wave' + str(waveno) + '/'
	for idx in idx_feature:

		savepath = basefolder + 'output/wave' + str(waveno) + '/' + str(idx) + '/'
		if not os.path.exists(savepath+'/Si.txt'):

			global_sobol_sensitivity_analysis.global_sobol_sensitivity_analysis(loadpath,idx,savepath)

	# ================================================================
    # Param ranking -Total effects
    # ================================================================
	loadpath = basefolder + 'data/wave' + str(waveno) + '/'
	loadpath_sobol = basefolder + 'output/wave' + str(waveno) + '/'
	gsa_parameters_ranking.gsa_parameters_ranking_S(loadpath,
												    loadpath_sobol,
												    gsa_mode="STi",
												    mode="max")

	# ================================================================
    # Param ranking -1st order effects
    # ================================================================
	loadpath = basefolder + 'data/wave' + str(waveno) + '/'
	loadpath_sobol = basefolder + 'output/wave' + str(waveno) + '/'
	gsa_parameters_ranking.gsa_parameters_ranking_S(loadpath,
												    loadpath_sobol,
												    gsa_mode="Si",
												    mode="max")

	# ================================================================
    # PLOT
    # ================================================================

	loadpath = basefolder + 'data/wave' + str(waveno) + '/'

	X = np.loadtxt(loadpath+'X.txt')
	Y = np.loadtxt(loadpath+'Y.txt')

	xlabels = read_labels_from_file(loadpath + "xlabels.txt")
	ylabels = read_labels_from_file(loadpath + "ylabels.txt")

	xlabels_latex = read_labels_from_file(loadpath + "xlabels_latex.txt")

	plotpath = basefolder + 'figures/wave' + str(waveno) + '/'

	os.system("mkdir -p "+plotpath)

	plot_dataset(X, Y, xlabels, ylabels, plotpath + "X_vs_Y.png")

	gpepath = basefolder + 'output/wave' + str(waveno) + '/0/'
	X = np.loadtxt(gpepath+'X_train.txt')

	Y = np.zeros((len(X),1),dtype=float)
	for idx in idx_feature:
		gpepath = basefolder + 'output/wave' + str(waveno) + '/' + str(idx) + '/'
		y = np.loadtxt(gpepath+'y_train.txt')
		Y = np.concatenate((Y,y.reshape(len(y),1)),axis=1)

	Y = np.delete(Y, 0, 1)

	plot_dataset(X, Y, xlabels, ylabels, plotpath + "X_vs_Y_train.png")

	# gpepath = basefolder + 'output/0/best_split/'
	# X = np.loadtxt(gpepath+'X_train.txt')

	# # Y = np.zeros((len(X),1),dtype=float)
	# for idx in idx_feature:
	# 	gpepath = basefolder + 'output/' + str(idx) + '/best_split/'
	# 	y = np.loadtxt(gpepath+'y_train.txt')
	# 	Y = np.concatenate((Y,y.reshape(len(y),1)),axis=1)

	# Y = np.delete(Y, 0, 1)

	# plot_dataset(X, Y, xlabels, ylabels, plotpath + "X_vs_Y_train_bestsplit.png")

	
	gsapath = basefolder + 'output/wave' + str(waveno) + '/'
	ST_all = np.zeros((len(xlabels),len(ylabels)),dtype=float)
	S1_all = np.zeros((len(xlabels),len(ylabels)),dtype=float)
	for i,idx in enumerate(idx_feature):
		ST = np.loadtxt(gsapath+str(idx)+'/STi.txt')
		S1 = np.loadtxt(gsapath+str(idx)+'/Si.txt')
		ST_all[:,i] = np.mean(ST, axis=0)
		S1_all[:,i] = np.mean(S1, axis=0)

	gsa_heat(ST_all, S1_all, xlabels, ylabels, plotpath, correction=False,horizontal=True, xlabels_latex=xlabels_latex)

	plot_rank_GSA(loadpath,
				  loadpath_sobol,
				  criterion="STi",
				  mode="max",
				  figname=plotpath+"Rank_max_test.png",
				  th=0.0,
				  separate_colors=False,
				  xlabels_latex=xlabels_latex)

	plot_rank_GSA(loadpath,
				  loadpath_sobol,
				  criterion="Si",
				  mode="max",
				  figname=plotpath+"Rank_max_1storder_test.png",
				  th=0.0,
				  separate_colors=True,
				  xlabels_latex=xlabels_latex)

if __name__ == '__main__':
	main()
