

import numpy as np
import torch

import json
import os
import pandas as pd
from copy import deepcopy


emulator_array = []
mean_list = []
std_list = []
iseScores = []

# loading input parameters
inputs = []
outputs = []
sim_ids = []

mesh_id = '0012'

out_path = '/home/tmb119/Dropbox/code/GPErks_library/p{}_ani/observed'.format(mesh_id)

cmd = "mkdir -p {}".format(out_path)
os.system(cmd)

observed_peaks = []
observed_errors = []

mesh_path = '/home/tmb119/Dropbox/portugal_data/P-{}/FinalMesh/motiontracking_endo'.format(mesh_id)


features = np.array(['ES_disp_global','ES_disp_anterior', 'ES_disp_posterior', 'ES_disp_septum', 'ES_disp_lateral', 'ES_disp_roof','ES_vol'])
#features = np.array(['ES_disp_global','ES_vol'])
for feature_index,feature in enumerate(features):

    json_settings_observed = "{}/output_data.json".format(mesh_path)
    f_input = open(json_settings_observed,"r")
    parameter_dict_observed = json.load(f_input)
    f_input.close()
    observed_peaks.append(parameter_dict_observed['{}'.format(feature)])

regions = np.array(['global','anterior', 'posterior', 'septum', 'lateral', 'roof'])
#regions = np.array(['global'])

for region_index,region in enumerate(regions):

    observed_errors.append((parameter_dict_observed['disp_error_{}'.format(region)]))

    #observed_errors = [2,2,2,2,2,2]

vol_error = 0.05 * parameter_dict_observed['ES_vol']
observed_errors.append(vol_error)

np.savetxt(out_path + "/exp_mean.txt",observed_peaks, fmt ='%1.4f', delimiter='\n')
np.savetxt(out_path + "/exp_std.txt",observed_errors, fmt = '%1.4f', delimiter='\n')

