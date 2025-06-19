

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

mesh_id = '0001'

waveno = 1

basefolder = './p' + mesh_id + '_ani/'

out_path = basefolder + 'data'

cmd = 'mkdir -p ' + out_path
os.system(cmd)


file_path = '/media/tmb119/tmgb/portugal_sims/p{}/ani/wave{}/'.format(mesh_id, waveno)
param_path = '/home/tmb119/Dropbox/portugal_gsa_data/submission_files/P-{}/wave{}/parameter_values'.format(mesh_id, waveno)

# file_path = '/media/tmb119/tmgb/post_process/ctcrt{}/wave{}/'.format(mesh_id, waveno)
# param_path = '/home/tmb119/Dropbox/gsa_data/submission_files/CTCRT{}/wave{}/parameter_values'.format(mesh_id, waveno)

###creating new targets for verification
# file_path = '/media/tmb119/Elements/Tom2_sims/ctcrt{}_ani/verification/samples/'.format(mesh_id, waveno)
# param_path = '/home/tmb119/Dropbox/gsa_data/submission_files/CTCRT{}/verification/parameter_values'.format(mesh_id, waveno)

endtimes = np.loadtxt('{}/sim_endtime.txt'.format(file_path), dtype=float)
# endtimes = np.loadtxt('/media/tmb119/Elements2/Tom2_sims/ctcrt{}_ani/wave{}/sim_endtime.txt'.format(mesh_id, waveno), dtype=float)
final_time = 1500
n_training_pts = 600
n_samples = 350
for pt in range(1,n_training_pts+1):
    sim_path = '{}/sim_{}'.format(file_path, pt)
    # sim_path = '{}/POSTPROC_DIR_{}'.format(file_path, pt)
    # orig_sim_path = '/media/tmb119/Elements2/Tom2_sims/ctcrt{}_ani/wave{}/sim_{}'.format(mesh_id, waveno, pt)

    check_exists = os.path.exists(sim_path)
    # check_exists = os.path.exists(orig_sim_path)

    if check_exists == True:
        
        if endtimes[pt-1] >= final_time:
            
            #print('simulation {} path exists'.format(pt))

            json_settings_inputparams_par = "{}/parameter_data_{}.json".format(param_path,pt)
            f_input = open(json_settings_inputparams_par,"r")
            input_dict = json.load(f_input)
            f_input.close()

            #input_dict['guccione_scaling_bulk'] = input_dict['guccione_scaling_bulk_anterior']
            #input_dict['guccione_scaling_ani'] = input_dict['guccione_scaling_ani_anterior']
            
            #regions = np.array(['anterior','posterior', 'septum', 'lateral', 'roof'])
            #for region in regions:
                #input_dict['guccione_scaling_bulk_{}'.format(region)]
                #input_dict['guccione_scaling_ani_{}'.format(region)]

            #inputs.append(parameter_dict_inputparams_par)

#             json_settings_inputparams_peri = "{}/pericardium_data_{}.json".format(param_path,pt)
#             f_input = open(json_settings_inputparams_peri,"r")
#             parameter_dict_inputparams_peri = json.load(f_input)
#             f_input.close()

#             input_dict = {**parameter_dict_inputparams_par, **parameter_dict_inputparams_peri}

            inputs.append(input_dict)

            json_settings_outputparams = "{}/output_data_{}.json".format(sim_path,pt)
            # json_settings_outputparams = "{}/strain_data.json".format(sim_path,pt)
            f_input = open(json_settings_outputparams,"r")
            parameter_dict_outputparams = json.load(f_input)
            f_input.close()
            outputs.append(parameter_dict_outputparams)

            # json_settings_2 = "{}/output_data_{}.json".format(orig_sim_path,pt)
            # f_input = open(json_settings_2,"r")
            # parameter_dict_2 = json.load(f_input)
            # f_input.close()

            # parameter_dict_outputparams['ES_vol'] = parameter_dict_2['ES_vol']
            # # outputs.append(parameter_dict_2)

            # outputs.append(parameter_dict_outputparams)

            # final_param_dict = {**parameter_dict_outputparams, **parameter_dict_2}

            sim_ids.append(pt)
            # outputs.append(final_param_dict)
            
print(outputs[0])
print(np.shape(sim_ids))

print(np.shape(inputs))
df_input = pd.DataFrame(inputs[:n_samples])
print(df_input.shape)
print(df_input.head())

features = np.array(['ES_disp_global','ES_disp_anterior', 'ES_disp_posterior', 'ES_disp_septum', 'ES_disp_lateral', 'ES_disp_roof','ES_vol'])
# features = np.array(['principal_strain_global','principal_strain_anterior', 'principal_strain_posterior', 'principal_strain_septum', 'principal_strain_lateral', 'principal_strain_roof', 'ES_vol'])
#features = np.array(['ES_disp_global','ES_vol'])
df_output = pd.DataFrame(outputs[:n_samples])
print(df_output.shape)
print(df_output.head())

df_y = df_output[features]
print(df_y.head())
print(df_y.shape)

# exp_mean = df_y.mean(axis=0)
# exp_std = df_y.std(axis=0)


if waveno == 1:

    cmd = 'mkdir -p ' + out_path + "/wave" +str(waveno)
    os.system(cmd)

    df_input.to_csv(out_path + "/wave" + str(waveno) + "/X.txt", float_format='%1.4f', index=False, header=False, sep = ' ')
    df_y.to_csv(out_path + "/wave" + str(waveno) + "/Y.txt", float_format='%1.4f', index=False, header=False, sep = ' ')

    np.savetxt(out_path + "/wave" + str(waveno) + "/xlabels.txt",df_input.columns, fmt ='%s', delimiter='\n')
    np.savetxt(out_path + "/wave" + str(waveno) + "/ylabels.txt",df_y.columns, fmt = '%s', delimiter='\n')
    np.savetxt(out_path + "/wave" + str(waveno) + "/features_idx_list.txt",np.arange(len(features)), fmt = '%i', delimiter='\n')
    np.savetxt(out_path +  "/wave" + str(waveno) + "/features_idx_list_hm.txt",np.arange(len(features)), fmt = '%i', delimiter='\n')

    np.savetxt(out_path +  "/wave" + str(waveno) + "/X_sims_ids.txt",sim_ids, fmt = '%i', delimiter='\n')
    ###creating new targets for verification
    # np.savetxt(out_path + "/exp_mean.txt", exp_mean.values, fmt = '%1.4f', delimiter='\n')
    # np.savetxt(out_path + "/exp_std.txt", exp_std.values, fmt = '%1.4f', delimiter='\n')
    # print(exp_mean.values)
    # print(exp_std.values)



else:

    cmd = 'mkdir -p ' + out_path + "/wave" +str(waveno)
    os.system(cmd)

    df_input.to_csv(out_path + "/wave" + str(waveno) + "/X_sims.txt", float_format='%1.4f', index=False, header=False, sep = ' ')
    df_y.to_csv(out_path + "/wave" + str(waveno) + "/Y_sims.txt", float_format='%1.4f', index=False, header=False, sep = ' ')

    # df_input_combined = deepcopy(df_input)
    # df_y_combined = deepcopy(df_y)
    
    # for wave in range(1,waveno):

    X_train = np.loadtxt(basefolder + 'data/wave' + str(waveno-1) + '/X.txt', dtype=float)
    Y_train = np.loadtxt(basefolder + 'data/wave' + str(waveno-1) + '/Y.txt', dtype=float)
    # Y_train = np.reshape(Y_train,(len(Y_train), 1))
    # for i in range(1,len(features)):
    #     Y_temp = np.loadtxt(basefolder + 'output/wave' + str(wave) + '/' + str(i) + '/y_train.txt', dtype=float)
    #     Y_train = np.concatenate((Y_train, Y_temp.reshape(len(Y_temp),1)), axis=1)

    df_Xtrain = pd.DataFrame(X_train, columns=df_input.columns, dtype=float)
    df_Ytrain = pd.DataFrame(Y_train, columns=df_y.columns, dtype=float)

    df_input_combined = pd.concat([df_Xtrain, df_input], ignore_index=True, sort=False)
    df_y_combined = pd.concat([df_Ytrain, df_y], ignore_index=True, sort=False)

    df_input_combined.to_csv(out_path +  "/wave" + str(waveno) + "/X.txt", float_format='%1.4f', index=False, header=False, sep = ' ')
    df_y_combined.to_csv(out_path + "/wave" + str(waveno) + "/Y.txt", float_format='%1.4f', index=False, header=False, sep = ' ')

    print(df_input_combined.head())
    print(df_y_combined.head())

    np.savetxt(out_path +  "/wave" + str(waveno) + "/xlabels.txt",df_input.columns, fmt ='%s', delimiter='\n')
    np.savetxt(out_path +  "/wave" + str(waveno) + "/ylabels.txt",df_y.columns, fmt = '%s', delimiter='\n')

    np.savetxt(out_path +  "/wave" + str(waveno) + "/features_idx_list.txt",np.arange(len(features)), fmt = '%i', delimiter='\n')
    np.savetxt(out_path +  "/wave" + str(waveno) + "/features_idx_list_hm.txt",np.arange(len(features)), fmt = '%i', delimiter='\n')
    np.savetxt(out_path +  "/wave" + str(waveno) + "/X_sims_ids.txt",sim_ids, fmt = '%i', delimiter='\n')


