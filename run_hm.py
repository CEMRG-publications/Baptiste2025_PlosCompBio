import os

import numpy as np

from GPErks.serialization.labels import read_labels_from_file
from GPErks_library import history_matching

from GPErks_library import HM_utils
from GPErks_library.plotting import *
from GPErks_library.hm_plotting import * 

def main():

	basefolder = './p0001_ani/'

	waveno = 1


	datapath = basefolder + 'data/wave'+ str(waveno) + '/'

	matchpath = basefolder + 'observed/'

	gpepath = basefolder + 'output/wave' + str(waveno) + '/'
	
	savepath = basefolder + 'hm_output/wave' + str(waveno) + '/'

	cutoff = 4.5
	
	# ================================================================
    # RUN HISTORY MATCHING
    # ================================================================

	# if not os.path.exists(savepath+"gpe.pth"):
	cmd = 'mkdir -p ' + savepath
	os.system(cmd)

	if waveno==1:
		prev_wave=''
	else:
		prev_wave=basefolder + 'hm_output/wave' + str(waveno-1) + '/'
		print(prev_wave)

	history_matching.history_matching(datapath,
				                     matchpath,
				                     gpepath,
				                     cutoff,
				                     waveno,
				                     savepath,
				                     previouswave=prev_wave,
				                     n_simuls=500,
				                     n_tests=100000,
				                     gpe_mode="full",
				                     features_list_file=None,
				                     X_test_file=None,
				                     output_X=True,
				                     maxno=1)


	# ================================================================
    # PLOT
    # ================================================================

	waves = [1]
	wavepath = basefolder + 'hm_output/'
	xlabels = read_labels_from_file(datapath + "xlabels_latex.txt")

	plot_inputspace(xlabels,
					outname=basefolder + "/figures/wave" + str(waveno) + "/training_input_distribution_plot.png",
					loadpath=basefolder,
					waveno=waveno,
					gpe_mode='full')
	
	plot_inputspace(xlabels,
					outname=basefolder + "/figures/wave" + str(waveno) + "/input_distribution_plot.png",
					loadpath=basefolder,
					waveno=waveno,
					gpe_mode='orig')

	plot_waves_paramSpace(wavepath,
                          waves,
                          xlabels,
                          figname=wavepath + "/wave" + str(waveno) + "/NIMP_regions",
                          idx_param=None,
                          Ncolors=None)


	plot_wave_full(wavepath+"/wave"+str(waves[-1])+"/wave_"+str(waves[-1]),
					xlabels=xlabels,
					display="impl",
					filename=wavepath+"/wave"+str(waves[-1])+"_impl"
					)

	plot_wave_full(wavepath+"/wave"+str(waves[-1])+"/wave_"+str(waves[-1]),
					xlabels=xlabels,
					display="var",
					filename=wavepath+"/wave"+str(waves[-1])+"_var"
					)

if __name__ == '__main__':
	main()
