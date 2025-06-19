import os

import numpy as np

from GPErks.serialization.labels import read_labels_from_file
from GPErks_library import history_matching

from GPErks_library import HM_utils
from GPErks_library.figures import *


def main():
    
    basefolder = './ctcrt29_ani/'
    waveno = 1
    waves = [1,2,3,4]

    wavepath = basefolder + 'data/'

    outpath = basefolder + "/figures/"

    
    xlabels = read_labels_from_file(wavepath + "wave" + str(waves[-2]) + "/xlabels_latex.txt")
    ylabels = read_labels_from_file(wavepath + "wave" + str(waves[-2]) + "/ylabels_latex.txt")
    
    # # # # plot_input_waves_boxplot(waves,
    # # # #                           wavefolder=wavepath,
    # # # #                           xlabels=xlabels,
    # # # #                           input_labels=np.arange(len(xlabels)),
    # # # #                           figname="boxplot_inputs",
    # # # #                           outpath=outfolder)
    
    plot_output_waves_boxplot(waves,
                            wavefolder=wavepath,
                            observedfolder=basefolder + "observed/",
                            outpath=outpath,
                            ylabels=ylabels,
                            feature_idx=np.arange(len(ylabels)),
                            figname="boxplot_outputs")

    # cases = ['29', '28', '20', '24','17', '02','01','12','15','30']
    # finalwavenos = [5, 9, 6, 1, 2, 1,1,1,1,4]


    # plot_diff_from_target(cases,
    #                 finalwavenos,
    #                 outpath=outfolder,
    #                 xlabels=xlabels,
    #                 input_labels=np.arange(9),
    #                 figure_size = (15.0,10.0),
    #                 plot_size = (3,3),
    #                 figname="verification_diff")

    # plot_mcmc_values(cases,
    #                 finalwavenos,
    #                 outpath=outpath,
    #                 xlabels=xlabels,
    #                 input_labels=np.arange(9),
    #                 figure_size = (15.0,10.0),
    #                 plot_size = (3,3),
    #                 figname="max_likelihood_inputs")
    
    # plot_errors(cases,
    #             finalwavenos,
    #             outpath=outpath,
    #             ylabels=ylabels,
    #             figure_size = (15.0,10.0),
    #             plot_size = (3,3),
    #             figname="errors_boxplot")

    # # plot_regional_stiffness_boxplot(cases=cases,
    # #                             finalwaveno=5,
    # #                             xlabels=xlabels,
    # #                             input_labels=np.arange(5),
    # #                             figname="boxplot_stiffnesses",
    # #                             outpath=outfolder)

    # observedfolder= "/home/tmb119/Dropbox/data/CTCRT24/FinalMesh"
    # simfolder="/media/tmb119/Elements/Tom2_sims/ctcrt24_ani/"
    

    # plot_sim_output_transient(observedfolder,
    #                           simfolder,
    #                           outpath=outfolder,
    #                          finalwaveno=2,
    #                          sim_ids=(1,363),
    #                          plot_size=(2,3),
    #                          figure_size=(15.0,10.0),
    #                          figname='output_transients')


    # plot_sim_volume_transient(observedfolder,
    #                           simfolder,
    #                           outpath=outfolder,
    #                          finalwaveno=2,
    #                          sim_ids=(1,363),
    #                          figure_size=(8.0,10.0),
    #                          figname='volume_transient')

    
    # cases = ['29', '28','20','24', '17', '02', '01', '12', '15', '30']
    # finalwavenos = [3, 8, 5 ,3, 6, 3, 4, 3, 4, 4]
    # # outpath = "/home/tmb119/Dropbox/figures/"

    # outpath = "/home/tmb119/Dropbox/figures/"
    # basefolder = './ctcrt28_ani/'
    # waveno = 8
    # xlabels = read_labels_from_file(basefolder + "data/wave" + str(waveno) + "/xlabels_latex.txt")
    # loadpath = basefolder + 'data/wave' + str(waveno) + '/'
    # loadpath_sobol = basefolder + 'output/wave' + str(waveno) + '/'

    # plot_mcmc_values(cases,
    #                 finalwavenos,
    #                 outpath=outpath,
    #                 xlabels=xlabels,
    #                 input_labels=np.arange(9),
    #                 figure_size = (15.0,10.0),
    #                 plot_size = (3,3),
    #                 figname="max_likelihood_inputs")
    
    # compare_bulk_ani(outpath=outpath,
	# 			  criterion="STi",
	# 			  mode="max",
	# 			  th=0.0,
    #               cases=cases)
    
    # plot_avg_heat(xlabels,
    #               loadpath,
    #                  outpath,
	# 			  loadpath_sobol,
	# 			  criterion="STi",
	# 			  mode="max",
	# 			  th=0.0,
    #               cases=cases)

if __name__ == '__main__':
	main()
