import os
import numpy as np
import json
import pandas as pd

from GPErks.serialization.labels import read_labels_from_file



# ================================================================
# GENERATING NEXT WAVE PARAMETER FILES
# ================================================================

meshid = '0012'
waveno = 2
basefolder = './p{}_ani/'.format(meshid)

loadpath = basefolder + 'data/wave1/'

xlabels = read_labels_from_file(loadpath + "xlabels.txt")

simspath = basefolder + 'hm_output/wave' +str(waveno-1)+ '/'

Xsimuls = np.loadtxt(simspath+'X_simul_'+ str(waveno-1)+ '.txt')


if "guccione_scaling_ani" in xlabels:
	# print('here')
	# outputpath = "/home/tmb119/Dropbox/gsa_data/submission_files/CTCRT" + meshid + "/global/wave" + str(waveno) + "/parameter_values/"
	outputpath = "/home/tmb119/Dropbox/portugal_gsa_data/submission_files/P-" + meshid + "/global/wave" + str(waveno) + "/parameter_values/"
	os.system("mkdir -p "+outputpath)

	# print(outputpath)

	for i in range(Xsimuls.shape[0]):
		values = np.asarray(Xsimuls[i])
		values[0] = round(values[0],6)
		values[1] = round(values[1],4)
		values[2] = round(values[2],4)
		values[3] = round(values[3],8)
		values[4] = round(values[4],4)

		parameter_data = dict(zip(xlabels, values))

		with open(outputpath +'parameter_data_{}.json'.format(i+1), 'w') as outfile:
			json.dump(parameter_data, outfile, indent=4)

elif "guccione_scaling_bulk_anterior" in xlabels:

	outputpath = "/home/tmb119/Dropbox/gsa_data/submission_files/CTCRT" + meshid + "/vary_bulk/wave" + str(waveno) + "/parameter_values/"
	os.system("mkdir -p "+outputpath)

	for i in range(Xsimuls.shape[0]):
		values = np.asarray(Xsimuls[i])
		values[0] = round(values[0],4)
		values[1] = round(values[1],6)
		values[2] = round(values[2],4)
		values[3] = round(values[3],6)
		values[4] = round(values[4],4)
		values[5] = round(values[5],6)
		values[6] = round(values[6],4)
		values[7] = round(values[7],6)
		values[8] = round(values[8],4)
		values[9] = round(values[9],6)
		values[10] = round(values[10],4)
		values[11] = round(values[11],4)
		values[12] = round(values[12],8)
		values[13] = round(values[13],4)

		parameter_data = dict(zip(xlabels, values))

		with open(outputpath +'parameter_data_{}.json'.format(i+1), 'w') as outfile:
			json.dump(parameter_data, outfile, indent=4)

else:

	# outputpath = "/home/tmb119/Dropbox/gsa_data/submission_files/CTCRT" + meshid + "/uniform/wave" + str(waveno) + "/parameter_values/"
	outputpath = "/home/tmb119/Dropbox/portugal_gsa_data/submission_files/P-" + meshid + "/wave" + str(waveno) + "/parameter_values/"
	os.system("mkdir -p "+outputpath)

	for i in range(Xsimuls.shape[0]):
		values = np.asarray(Xsimuls[i])
		values[0] = round(values[0],6)
		values[1] = round(values[1],6)
		values[2] = round(values[2],6)
		values[3] = round(values[3],6)
		values[4] = round(values[4],6)
		values[5] = round(values[5],4)
		values[6] = round(values[6],4)
		values[7] = round(values[7],8)
		values[8] = round(values[8],4)

		parameter_data = dict(zip(xlabels, values))

		with open(outputpath +'parameter_data_{}.json'.format(i+1), 'w') as outfile:
			json.dump(parameter_data, outfile, indent=4)
