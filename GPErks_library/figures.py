import re
import numpy as np
import pandas as pd
import seaborn as sns
import os
import emcee
import corner

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib import ticker

from GPErks.serialization.labels import read_labels_from_file


def plot_mcmc_posterior(cases,
                    finalwavenos,
                    outpath,
                    ylabels,
                    input_labels=[0],
                    figure_size = (15.0,10.0),
                    plot_size = (3,3),
                    figname="errors_boxplot"):
    

    fig = plt.figure(figsize = figure_size)

    #colors =  ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    df = pd.DataFrame()
    #
    ax = plt.subplot(111)
    for feature_no, feature in enumerate(ylabels):
         
        ax = plt.subplot(plot_size[0], plot_size[1], i+1)

        data = []

        for i, case in enumerate(cases):

            print('case:', case)
            basefolder = "./ctcrt" + case + "_ani/"
            mcmcfolder = basefolder + "mcmc_output/"  + "wave" + str(finalwavenos[i]) + "/"

            print('loading mcmc samples...')
            filename = mcmcfolder + "samples.h5"
            reader = emcee.backends.HDFBackend(filename)

            burnin = 5000
            thin = 10

            flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)

            xlabels = read_labels_from_file(datapath + "xlabels.txt")
            ylabels = read_labels_from_file(datapath + "ylabels.txt")

    fig = corner.corner(
        flat_samples, labels=xlabels, label_kwargs=dict(fontsize=18),color=cm.RdPu(0.95)
    );        
    df[feature] = data

    ax= df.boxplot()
    ax.set_xlabel("Output features")
    plt.savefig(outpath + figname)
    plt.show()

def plot_errors(cases,
                    finalwavenos,
                    outpath,
                    ylabels,
                    input_labels=[0],
                    figure_size = (15.0,10.0),
                    plot_size = (3,3),
                    figname="errors_boxplot"):
    

    fig = plt.figure(figsize = figure_size)

    #colors =  ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    df = pd.DataFrame()
    #
    ax = plt.subplot(111)
    for feature_no, feature in enumerate(ylabels):

        data = []

        for i, case in enumerate(cases):

            print('case:', case)
            basefolder = "./ctcrt" + case + "_ani/"
            valuesfolder = basefolder + "mcmc_output/"

            input = np.loadtxt(valuesfolder + "wave" + str(finalwavenos[i]) + "/posterior_errors.txt")
            
            data.append(input[feature_no])
            # all_cases_std.append(input[1,:])
        
        df[feature] = data

    ax= df.boxplot()
    ax.set_xlabel("Output features")
    plt.savefig(outpath + figname)
    plt.show()


def plot_mcmc_values(cases,
                    finalwavenos,
                    outpath,
                    xlabels,
                    input_labels=[0],
                    figure_size = (9.0,3.0),
                    plot_size = (3,3),
                    figname="max_likelihood_inputs"):

        
    afont = {'fontname':'Arial'}
    markers_filled = ['o', 's', 'D', 'v', '^', '<','>','p','h','H']
    ## markers = ['*', '+','.','2', 'D', 'h', 'o', 's','x', 0]

    fig = plt.figure(figsize = figure_size)

    colours = ['#1D4273','#537C97','#C6BEB5','#B67A84','#7A165B']

    ##colors =  ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']

    #

    all_cases_x = []
    all_cases_std = []

    for i, case in enumerate(cases):
        print('case:', case)
        basefolder = "./ctcrt" + case + "_ani/"
        valuesfolder = basefolder + "mcmc_output/"

        input = np.loadtxt(valuesfolder + "wave" + str(finalwavenos[i]) + "/max_prob_values_NROY.txt")
        
        all_cases_x.append(input[0,0:5])

    print(all_cases_x)

    df = pd.DataFrame(data=all_cases_x,columns=xlabels[:5] )
    

    # print(df)
    print(df.median(axis=0))

    ax = sns.boxplot(df,  palette = colours, fill=False, linewidth=5)
    ax = sns.lineplot(data=df.T, palette=['#bdc3c7']*10, alpha=1, legend=False, dashes=[(4, 2)] * 10, markers = markers_filled, markersize=15)
    ax.set_ylabel('Stiffness',**afont, fontsize=30)
    ax.set_xticklabels(xlabels[:5],**afont, fontsize=30)
    ax.set_yticklabels(np.arange(-0.5,4.5,0.5),**afont, fontsize=24)
    plt.savefig(outpath + figname)

    ## for i, input_label in enumerate(input_labels):

    ##     ax = plt.subplot(plot_size[0], plot_size[1], i+1)
    ##     for case_no in range(len(cases)):
    ##         # ax.errorbar(case_no+1,all_cases_x[case_no][i], yerr = all_cases_x[case_no][i]/3, xerr = 0, ecolor = cm.RdPu(0.95), elinewidth=5, capsize=3, dash_capstyle='round',zorder =0, alpha=0.50) #95a5a6
    ##         ax.scatter(case_no+1,all_cases_x[case_no][i], s=40, color=cm.RdPu(0.95), zorder=2)
    ##         #ax.errorbar(xval, yval, xerr = 0.4, yerr = 0.5)
    ##         ax.set_xlabel("patient case") 
    ##         ax.set_ylabel("{}".format(xlabels[input_label])) 

    ##     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
    ##     #ax.locator_params(axis='x', nbins=4)
    ##     ax.set_xticklabels(np.arange(1,len(cases)+1))
    ##     ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,len(cases)+1)))
    ##     if i >=0 and i<=4:
    ##         ax.set_ylim((0,6))


    plt.savefig(outpath + figname)

 
    




def plot_input_waves_boxplot(waves,
                              wavefolder,
                              outpath,
                              xlabels,
                              input_labels=[0],
                              figure_size = (15.0,10.0),
                              plot_size = (3,3),
                              figname="boxplot_inputs"):
    
    
    fig = plt.figure(figsize = figure_size)

    Ncolors = len(waves)+1

    evenly_spaced_interval = np.linspace(0.2, 1, Ncolors)
    colors = [cm.RdPu(x) for x in evenly_spaced_interval]

    for i, input_label in enumerate(input_labels):

        all_waves = []

        for wave in waves:
            input = np.loadtxt(wavefolder + "wave" + str(wave) + "/X.txt", usecols=input_label)
            if wave > 1:
                input = np.loadtxt(wavefolder + "wave" + str(wave) + "/X_sims.txt", usecols=input_label)
            all_waves.append(input)
        # print(len(all_waves[0]))
        # print(len(all_waves[1]))
        # print(len(all_waves[2]))      
        
        ax = plt.subplot(plot_size[0], plot_size[1], i+1)
        ax = sns.boxplot(data=all_waves, palette=colors, fill=False)
        # ax = sns.violinplot(data=all_waves)
        ax.set_xlabel("wave number") 
        ax.set_ylabel("{}".format(xlabels[input_label])) 
        ax.set_xticklabels(waves)
    
    plt.savefig(outpath + figname)


def plot_regional_stiffness_boxplot(finalwaveno,
                              cases,
                              outpath,
                              xlabels,
                              input_labels=[0],
                              figure_size = (15.0,10.0),
                              plot_size = (2,3),
                              figname="boxplot_stiffnesses"):
    
    
    fig = plt.figure(figsize = figure_size)

    #colors =  ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    colours = ['#1D4273','#537C97','#C6BEB5','#B67A84','#7A165B']

    for i, input_label in enumerate(input_labels):

        all_cases = []

        for case in cases:
            basefolder = "./ctcrt" + case + "_ani/"
            wavefolder = basefolder + "data/"
            
            if finalwaveno > 1:
                input = np.loadtxt(wavefolder + "wave" + str(finalwaveno) + "/X_sims.txt", usecols=input_label)
            else:
                input = np.loadtxt(wavefolder + "wave" + str(finalwaveno) + "/X.txt", usecols=input_label)
            all_cases.append(input)
        # print(len(all_waves[0]))
        # print(len(all_waves[1]))
        # print(len(all_waves[2]))      
        
        ax = plt.subplot(plot_size[0], plot_size[1], i+1)
        ax = sns.boxplot(data=all_cases, color = colours[i], fill=False)
        # ax = sns.violinplot(data=all_waves)
        ax.set_xlabel("patient case") 
        ax.set_ylabel("{}".format(xlabels[input_label])) 
        ax.set_xticklabels(np.arange(1,len(cases)+1))
    
    plt.savefig(outpath + figname)




def plot_output_waves_boxplot(waves,
                              wavefolder,
                              observedfolder,
                              outpath,
                              ylabels,
                              feature_idx=[0],
                              figure_size=(16.0,14.0),
                              plot_size=(3,3),
                              figname="boxplot_outputs"):
    
    fig = plt.figure(figsize = figure_size)

    Ncolors = len(waves)+1

    afont = {'fontname':'Arial'}

    evenly_spaced_interval = np.linspace(0.2, 1, Ncolors)
    colors = [cm.RdPu(x) for x in evenly_spaced_interval]

    for idx in feature_idx:
    
        m = np.loadtxt(observedfolder + "exp_mean.txt", dtype=float)
        s = np.loadtxt(observedfolder + "exp_std.txt", dtype=float)

        feature_mean = m[idx]
        feature_std = s[idx]
        
        all_waves = []
        
        for wave in waves:
            feature = np.loadtxt(wavefolder + "wave" + str(wave) + "/Y.txt", usecols=idx)
            if wave > 1:
                feature = np.loadtxt(wavefolder + "wave" + str(wave) + "/Y_sims.txt", usecols=idx)
            all_waves.append(feature)

        # print(len(all_waves[0]))
        # print(len(all_waves[1]))
        # print(len(all_waves[2]))      
        
        ax = plt.subplot(plot_size[0], plot_size[1], idx+1)
        ax = sns.boxplot(data=all_waves, palette=colors, fill=False)
        # ax = sns.violinplot(data=all_waves)
        ax.add_patch(
            patches.Rectangle(
            xy=(-1, feature_mean-(2*feature_std)),  # point of origin.
            width=waves[-1]+1, height=4*feature_std, linewidth=1,
            color='#ecf0f1', fill=True, alpha=1, zorder=0))
        ax.hlines(y=feature_mean, xmin=-1, xmax=waves[-1], color='k', lw=1)

        waves_plot = [w - 1 for w in waves]
        ax.set_xlabel("wave number", **afont, fontsize=18) 
        ax.set_ylabel("{}".format(ylabels[idx]),**afont, fontsize=18) 
        ax.set_xticklabels(waves_plot, **afont, fontsize=18)
        ax.set_yticklabels([])
        ax.set_xlim([-1,waves[-1]])

    
    plt.savefig(outpath + figname)


def plot_sim_output_transient(observedfolder,
                              simfolder,
                              outpath,
                             finalwaveno=5,
                             sim_ids=(1,10),
                             plot_size=(2,3),
                             figure_size=(15.0,10.0),
                             figname='output_transients'):
      
    regions = np.array(['anterior', 'posterior', 'septum', 'lateral', 'roof'])
    #region_colours = ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    region_colours = ['#1D4273','#537C97','#C6BEB5','#B67A84','#7A165B']
      
    fig = plt.figure(figsize = figure_size)
    final_time = 1500
    ES_time = 90

    mesh_regions = np.loadtxt(observedfolder +'/new_tags.dat',dtype=int)
    obs_displacement_file = np.loadtxt(observedfolder +'/motiontracking_endo/mag_displacement_ECs.dat',dtype=float)
      
    obs_avg_global = np.mean(obs_displacement_file, axis = 0)
    time_phase = np.arange(0,101,10)
    sim_time_phase = np.arange(-5,71)
      
    for i in range(sim_ids[0],(sim_ids[1]+1)):
        
        file_path = simfolder+"wave1/sim_{}".format(i)
        check_exists = os.path.exists(file_path)
        if check_exists == True:
            
            sim_vol = np.loadtxt(file_path +'/LAendo.vol.dat'.format(i), dtype=float,skiprows=0)
            if sim_vol[-1,0] == final_time:
                
                displacement_file = np.loadtxt(file_path +'/temp/mag_displacement_ECs.dat',dtype=float)

                avg_global = np.mean(displacement_file, axis = 0)

                ax = plt.subplot(plot_size[0], plot_size[1], 1)

                ax.plot(sim_time_phase,avg_global[::2], color='#bdc3c7', alpha=0.15) #bdc3c7
                ax.set_title('global',fontsize=20)
                # ax.set_xlabel('Time Phase')
                # ax.set_ylabel('Average Endocardial Displacement (mm)')
                ax.scatter(40, avg_global[ES_time], marker='.', s=30, color='#bdc3c7')
                ax.set_ylim([0,20])
                ax.set_xlim([0,100]) 
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])


                for region_id, region in enumerate(regions):

                    ax = plt.subplot(plot_size[0], plot_size[1], region_id + 2)

                    region_list = np.where(mesh_regions == region_id+1)[0]
                    regional_disp = np.take(displacement_file,region_list,axis=0)
                    avg_regional = np.mean(regional_disp, axis = 0)
                    ax.plot(sim_time_phase,avg_regional[::2], color='#bdc3c7', alpha=0.15)
                    
                    #ax.plot(avg_regional, alpha=0.3, label = labels[idx] )
                    ax.set_title('{}'.format(regions[region_id]),fontsize=20)
                    # ax.set_xlabel('Time Phase')
                    # ax.set_ylabel('Average Endocardial Displacement (mm)')
                    ax.scatter(40, avg_regional[ES_time], marker='.', s=30, color='#bdc3c7')
                    ax.set_ylim([0,20])
                    ax.set_xlim([0,100])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
    
    for i in range(sim_ids[0],(sim_ids[1]+1)):
            
        file_path = simfolder+ "wave" + str(finalwaveno) + "/sim_{}".format(i)
        #print('here')
        check_exists = os.path.exists(file_path)
        if check_exists == True:
            
            sim_vol = np.loadtxt(file_path +'/LAendo.vol.dat'.format(i), dtype=float,skiprows=0)
            if sim_vol[-1,0] == final_time:
                
                displacement_file = np.loadtxt(file_path +'/temp/mag_displacement_ECs.dat',dtype=float)

                avg_global = np.mean(displacement_file, axis = 0)

                ax = plt.subplot(plot_size[0], plot_size[1], 1)

                ax.plot(sim_time_phase,avg_global[::2], color=cm.RdPu(1.0), alpha=0.3)
                ax.set_title('global',fontsize=20)
                # ax.set_xlabel('Time Phase')
                # ax.set_ylabel('Average Endocardial Displacement (mm)')
                ax.scatter(40, avg_global[ES_time], marker='.', s=30, color=cm.RdPu(1.0)) #34495e
                ax.set_ylim([0,20])
                ax.set_xlim([0,100])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])


                for region_id, region in enumerate(regions):

                    ax = plt.subplot(plot_size[0], plot_size[1], region_id + 2)

                    region_list = np.where(mesh_regions == region_id+1)[0]
                    regional_disp = np.take(displacement_file,region_list,axis=0)
                    avg_regional = np.mean(regional_disp, axis = 0)
                    ax.plot(sim_time_phase,avg_regional[::2], color=region_colours[region_id], alpha=0.3)
                    
                    #ax.plot(avg_regional, alpha=0.3, label = labels[idx] )
                    ax.set_title('{}'.format(regions[region_id]), fontsize=20)
                    # ax.set_xlabel('Time Phase')
                    # ax.set_ylabel('Average Endocardial Displacement (mm)')
                    ax.scatter(40, avg_regional[ES_time], marker='.', s=30, color=region_colours[region_id])
                    ax.set_ylim([0,20])
                    ax.set_xlim([0,100])
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])


    ax = plt.subplot(plot_size[0], plot_size[1], 1)

    ax.plot(time_phase,obs_avg_global, '#2c3e50', label='observed')
    ax.set_xlim([0,100])
    #ax.set_ylim(math.ceil(np.min(obs_avg_global))-2,math.ceil(np.max(obs_avg_global))+6)
    #ax.legend()
    #ax.grid()


    for region_id, region in enumerate(regions):

        ax = plt.subplot(plot_size[0], plot_size[1], region_id + 2)

        region_list = np.where(mesh_regions == region_id+1)[0]
        obs_regional_disp = np.take(obs_displacement_file,region_list,axis=0)
        obs_avg_regional = np.mean(obs_regional_disp, axis = 0)

        ax.plot(time_phase,obs_avg_regional, '#2c3e50', label='observed')
        ax.set_xlim([0,100])
        #ax.legend()
        #ax.grid()


    plt.savefig(outpath + "sim_disp_curves.png", bbox_inches="tight", dpi=300)


def plot_sim_volume_transient(observedfolder,
                              simfolder,
                              outpath,
                             finalwaveno=5,
                             sim_ids=(1,10),
                             figure_size=(8.0,10.0),
                             figname='volume_transient'):
      
    regions = np.array(['anterior', 'posterior', 'septum', 'lateral', 'roof'])
    #region_colours = ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    region_colours = ['#1D4273','#537C97','#C6BEB5','#B67A84','#7A165B']
      
    #ßfig = plt.figure(figsize = figure_size)
    fig,ax = plt.subplots(1,figsize = figure_size)
    final_time = 1500
    ES_time = 900

    # mesh_regions = np.loadtxt(observedfolder +'/new_tags.dat',dtype=int)
    # obs_displacement_file = np.loadtxt(observedfolder +'/motiontracking_endo/mag_displacement_ECs.dat',dtype=float)
      
    # obs_avg_global = np.mean(obs_displacement_file, axis = 0)
    time_phase = np.arange(0,101,10)
    sim_time_phase = np.arange(-5,71)
      
    for i in range(sim_ids[0],(sim_ids[1]+1)):
        
        file_path = simfolder+"wave1/sim_{}".format(i)
        check_exists = os.path.exists(file_path)
        if check_exists == True:
            
            sim_vol = np.loadtxt(file_path +'/LAendo.vol.dat'.format(i), dtype=float,skiprows=0)
            if sim_vol[-1,0] == final_time:
                
                #displacement_file = np.loadtxt(file_path +'/temp/mag_displacement_ECs.dat',dtype=float)

                #avg_global = np.mean(displacement_file, axis = 0)

                #ax = plt.subplots(1)

                ax.plot(sim_time_phase,sim_vol[0::20,1], color='#bdc3c7', alpha=0.15)
                #ax.set_title('global')
                ax.set_xlabel('Time Phase (%)')
                ax.set_ylabel('Endocardial Volume (ml)')
                ax.scatter(40, sim_vol[ES_time,1], marker='.', s=30, color='#bdc3c7')
                ax.set_ylim([0,200])
                ax.set_xlim([0,100])

    
    for i in range(sim_ids[0],(sim_ids[1]+1)):
            
        file_path = simfolder+ "wave" + str(finalwaveno) + "/sim_{}".format(i)
        #print('here')
        check_exists = os.path.exists(file_path)
        if check_exists == True:
            
            sim_vol = np.loadtxt(file_path +'/LAendo.vol.dat'.format(i), dtype=float,skiprows=0)
            if sim_vol[-1,0] == final_time:
                
                
                ax.plot(sim_time_phase,sim_vol[0::20,1], color=cm.RdPu(1.0), alpha=0.3)
                ax.scatter(40, sim_vol[ES_time,1], marker='.', s=30, color=cm.RdPu(1.0))
                ax.set_ylim([0,200])
                ax.set_xlim([0,100])


    obs_vol = np.loadtxt(observedfolder +'/motiontracking_closedendo/obs_endovolume.dat', dtype=float,skiprows=0)

    ax.plot(time_phase,obs_vol, '#2c3e50', label='observed')
    ax.set_xlim([0,100])
    #ax.set_ylim(math.ceil(np.min(obs_avg_global))-2,math.ceil(np.max(obs_avg_global))+6)
    #ax.legend()
    #ax.grid()


        #ax.grid()


    plt.savefig(outpath + "sim_vol_curves.png", bbox_inches="tight", dpi=300)

    
def plot_diff_from_target(cases,
                    finalwavenos,
                    outpath,
                    xlabels,
                    input_labels=[0],
                    figure_size = (15.0,10.0),
                    plot_size = (3,3),
                    figname="max_likelihood_inputs"):
    

    fig,ax = plt.subplots(1,figsize = figure_size)

    #colors =  ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']

    #

    all_cases_x = []
    all_cases_std = []

    for i, case in enumerate(cases):
        basefolder = "./ctcrt" + case + "_ani/"
        valuesfolder = basefolder + "mcmc_output/"

        input = np.loadtxt(valuesfolder + "wave" + str(finalwavenos[i]) + "/percentage_diff_from_target.txt")
        
        all_cases_x.append(input)
        # all_cases_std.append(input[1,:])

    print(all_cases_x)
    # print(all_cases_std)

    # print(all_cases_x[0][5])
    # print(all_cases_x[3][4])

    avg_diff = np.mean(all_cases_x, axis=0)
    std_diff = np.std(all_cases_x, axis=0)
    print(std_diff)

    # fig,ax = plt.subplots(1,figsize = (8.0,8.0))
    for i in range(len(avg_diff)):

        ax.errorbar(i+1,avg_diff[i], yerr = std_diff[i], xerr = 0, ecolor = cm.RdPu(0.95), elinewidth=5, capsize=3, dash_capstyle='round',zorder =0, alpha=0.50) #95a5a6
        ax.scatter(i+1,avg_diff[i], s=40, color=cm.RdPu(0.95), zorder=2)
        ax.set_xlabel("input parameters") 
        ax.set_ylabel("{}".format(input_labels[i])) 

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
        #ax.locator_params(axis='x', nbins=4)
        ax.set_xticklabels(np.arange(1,len(avg_diff)+1))
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,len(avg_diff)+1)))
    
    plt.show()
    plt.savefig(outpath + figname)


    # for i, input_label in enumerate(input_labels):

    #     ax = plt.subplot(plot_size[0], plot_size[1], i+1)
    #     for case_no in range(len(cases)):
    #         ax.errorbar(case_no+1,all_cases_x[case_no][i], yerr = all_cases_x[case_no][i]/3, xerr = 0, ecolor = cm.RdPu(0.95), elinewidth=5, capsize=3, dash_capstyle='round',zorder =0, alpha=0.50) #95a5a6
    #         ax.scatter(case_no+1,all_cases_x[case_no][i], s=40, color=cm.RdPu(0.95), zorder=2)
    #         #ax.errorbar(xval, yval, xerr = 0.4, yerr = 0.5)
    #         ax.set_xlabel("patient case") 
    #         ax.set_ylabel("{}".format(xlabels[input_label])) 

    #     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
    #     #ax.locator_params(axis='x', nbins=4)
    #     ax.set_xticklabels(np.arange(1,len(cases)+1))
    #     ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,len(cases)+1)))
    #     if i >=0 and i<=4:
    #         ax.set_ylim((0,6))


    # plt.savefig(outpath + figname)


def compare_bulk_ani(
                  outpath,
                  rank_file=None,
                  criterion="STi",
                  mode="max",
                  figname="",
                  normalise=False,
                  th=0.0,
                  annotate=False,
                  figsize=(15,5),
                  fontsize=14,
                  cases=[]):

    """
    Plots the parameter ranking for a .

    Args:
        - datapath: folder with data (xlabels.txt, etc...)
        - loadpath: path where you saved your parameter ranking 
        - rank_file: if you want to provide a different parameter ranking file 
                     that is not in the loadpath folder
        - criterion: STi or Si e.g. total or first order effects to use for ranking
        - mode: max or mean to rank the parameters
        - figname: name of output figure 
        - normalise: if you want to normalise so that the values all sum up to 1 
        - th: threshold to determine which parameters are important and which ones aren't
        - annotate: write numbers on top of each bar
        - figsize: size of output figure
        - fontsize: size of figure font
        - xlabels_latex: parameter names in latex for paper plots
        - separate_colors: if you want a different colour for important and unimportant parameters
        - color_important: what colour you want the important parameter bars to be
    """

    #color = ["#ff8000"]
    
    regions = np.array(['anterior', 'posterior', 'septum', 'lateral', 'roof'])
    #region_colours = ['#28337F','#54709E','#ACA8A2','#E07C59','#A10015']
    region_colours = ['#1D4273','#537C97','#C6BEB5','#B67A84','#7A165B']

    afont = {'fontname':'Arial'}
    fontsize = 16

    fig = plt.figure(figsize = (24.0,12.0)) 

    for i, region in enumerate(regions):
        ani_data = []
        bulk_data  = []

        for case in cases:
            print(case)

            basefolder = './ctcrt' + case + '/'
            waveno = 1
            
            loadpath = basefolder + 'output/wave' + str(waveno) + '/'

                
            if criterion == "Si":
                tag = "first-order"
            elif criterion == "STi":
                tag = "total"

            # if rank_file is None:
            rank_file = loadpath+"/Rank_"+criterion+"_"+mode+".txt"

            f = open(rank_file,"r")
            lines = f.readlines()

            r_dct = {}
            for line in lines:
                line_split = re.split(r'\t+', line)
                r_dct[line_split[0]] = float(line_split[1])

            del r_dct["Iz_2"]
            del r_dct["ED_pressure_kPa"]
            del r_dct["ES_pressure_kPa"]

            print(r_dct)
            
            ani_data.append(r_dct['guccione_scaling_ani_{}'.format(region)])
            bulk_data.append(r_dct['guccione_scaling_bulk_{}'.format(region)])

            # list_df = list(zip(bulk_data, ani_data))

            # # Assign data to tuples.
            # list_df


            # # Converting lists of tuples into
            # # pandas Dataframe.
            # df = pd.DataFrame(list_df,
            #                 columns=['C', 'alpha'])
            # df['case'] = [1]
            # print(df)

        ax = plt.subplot(2, 3, i+1)
        ax.bar(np.arange(len(cases))+1, bulk_data, color=cm.Blues(0.80), label="$C$")
        ax.bar(np.arange(len(cases))+1, ani_data, bottom=bulk_data, color=cm.RdPu(0.80),label=r"$\alpha$")
        ax.set_title(region, **afont,fontsize=fontsize)
        ax.set_xlabel("Case", **afont, fontsize=fontsize) 
        ax.set_ylabel("Sensitivity", **afont, fontsize=fontsize)
        ax.set_xticklabels(np.arange(0,11,2),**afont, fontsize=fontsize-2)
        ax.set_ylim([0,0.85])
        ax.set_yticklabels(np.around(np.arange(0,0.85,0.1),1),**afont, fontsize=fontsize-2)

        ax.legend()

    plt.savefig(outpath + "compare_bulk_ani.png", bbox_inches="tight", dpi=300)      
    plt.show()


    return r_dct


def plot_avg_heat(xlabels,
                  loadpath,
                  outpath,
                  rank_file=None,
                  criterion="STi",
                  mode="max",
                  figname="",
                  normalise=False,
                  th=0.0,
                  annotate=False,
                  figsize=(15,5),
                  fontsize=14,
                  cases=[]):

    """
    Plots the parameter ranking for a .

    Args:
        - datapath: folder with data (xlabels.txt, etc...)
        - loadpath: path where you saved your parameter ranking 
        - rank_file: if you want to provide a different parameter ranking file 
                     that is not in the loadpath folder
        - criterion: STi or Si e.g. total or first order effects to use for ranking
        - mode: max or mean to rank the parameters
        - figname: name of output figure 
        - normalise: if you want to normalise so that the values all sum up to 1 
        - th: threshold to determine which parameters are important and which ones aren't
        - annotate: write numbers on top of each bar
        - figsize: size of output figure
        - fontsize: size of figure font
        - xlabels_latex: parameter names in latex for paper plots
        - separate_colors: if you want a different colour for important and unimportant parameters
        - color_important: what colour you want the important parameter bars to be
    """

    #color = ["#ff8000"]


    fig = plt.figure(figsize = (25.0,15.0)) 

    # df = pd.DataFrame()
    df = pd.DataFrame(columns = xlabels)
 

    for case in cases:

        print(case)

        basefolder = './ctcrt' + case + '/'
        waveno = 1
        
        loadpath = basefolder + 'output/wave' + str(waveno) + '/'

            
        if criterion == "Si":
            tag = "first-order"
        elif criterion == "STi":
            tag = "total"

        # if rank_file is None:
        rank_file = loadpath+"/Rank_"+criterion+"_"+mode+".txt"

        f = open(rank_file,"r")
        lines = f.readlines()

        r_dct = {}
        for line in lines:
            line_split = re.split(r'\t+', line)
            r_dct[line_split[0]] = float(line_split[1])

        df = df.append(r_dct, ignore_index=True)
        # print(df.head())
        


    # print(df)
    avg = df.mean()
    np.savetxt(outpath + "/" + criterion + ".txt", avg, fmt="%.6f")
    print(avg)
        # parameters = list(r_dct.keys())
	# print(parameters)

    #     for parameter in parameters:
    #         df[parameter] = 
	#     param_ranges = [(settings[parameter]["lower_limit"], settings[parameter]["upper_limit"]) for parameter in parameters]

    #     del r_dct["Iz_2"]
    #     del r_dct["ED_pressure_kPa"]
    #     del r_dct["ES_pressure_kPa"]

    #     print(r_dct)
        
    #     ani_data.append(r_dct['guccione_scaling_ani_{}'.format(region)])
    #     bulk_data.append(r_dct['guccione_scaling_bulk_{}'.format(region)])

    #     # list_df = list(zip(bulk_data, ani_data))

    #     # # Assign data to tuples.
    #     # list_df


    #     # # Converting lists of tuples into
    #     # # pandas Dataframe.
    #     # df = pd.DataFrame(list_df,
    #     #                 columns=['C', 'alpha'])
    #     # df['case'] = [1]
    #     # print(df)

    #     ax = plt.subplot(2, 3, i+1)
    #     ax.bar(np.arange(len(cases))+1, bulk_data, color=cm.Blues(0.80), label="C")
    #     ax.bar(np.arange(len(cases))+1, ani_data, bottom=bulk_data, color=cm.RdPu(0.80),label="alpha")
    #     ax.set_title(region,fontsize=14)
    #     ax.set_xlabel("case", fontsize=14) 
    #     ax.set_ylabel("ST",fontsize=14) 
    #     ax.set_ylim([0,0.85])
    #     ax.legend()

    # plt.savefig(outpath + "compare_bulk_ani.png", bbox_inches="tight", dpi=300)      
    # plt.show()


    return df
