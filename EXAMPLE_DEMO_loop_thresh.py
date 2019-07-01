#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse UP and DOWN state duration from simulated data
Loop over different UP/DOWN state detection parameters to evaluate robustness

Used to produce Fig S3 of paper:
Nghiem T.-A.E. et al, https://www.biorxiv.org/content/10.1101/430405v3
"""
#%% Make spike trains, select clean cells

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import UP_read


len_state_all = [40, 50, 60, 70]
ratioThreshold_all = [0.01, 0.02, 0.1, 0.2]

pearson_allparams = np.zeros((len(len_state_all), len(ratioThreshold_all)))
pval_allparams = np.zeros((len(len_state_all), len(ratioThreshold_all)))

iilen = -1

for len_state in len_state_all:
    jjthresh = -1
    iilen += 1
    for ratioThreshold in ratioThreshold_all:
        jjthresh+=1
        #%% to collect statistics and plot
        # number of simulation files to analyse
        Nsims = 10
        
        # to record statistics in each file
        pearson_all = np.zeros(Nsims)
        pvalue_all = np.zeros(Nsims)
        pearson_shuffled = np.zeros(Nsims)
        err_shuffled = np.zeros(Nsims)
        
        # all UP-DOWN state durations
        dur_DOWN_all = []
        dur_UP_all = []
        
        # statistics of UP-DOWN state durations
        dur_UP_mean = np.zeros(Nsims)
        dur_UP_std = np.zeros(Nsims)
        dur_DOWN_mean = np.zeros(Nsims)
        dur_DOWN_std = np.zeros(Nsims)
        
        # for plotting
        cmap = plt.get_cmap('magma') 
        plt.rcParams.update({'font.size': 15})
        #%% network simulation parameters
        badap = 10 # mean adaptation strength
        noise_mean = 49 # mean noise amplitude
        noise_var = 3 # variance of noise amplitude across simulations
        noise_range = np.linspace(noise_mean - noise_var, 
                                  noise_mean + noise_var, Nsims)
        
        
        
        #%% loop over all simulation files, read UP-DOWN duration and compute pearson
        
        for isim in range(Nsims): # each file: different noise amplitude simulation
            filename = 'data_noise3_adap'+str(badap)+'/example_noise'+str(isim)+'.npy'
            
            data = np.load(filename)
            
            len_transient = 10000 # transient to discard, in time steps
            train_cut = 0.8*data[1][len_transient:] + 0.2*data[2][len_transient:]
            
            # spike train to compute UP state firing rate
            train_rate_all = train_cut
            
            train_rate_E = data[1][len_transient:]
            
            train_rate_I = data[2][len_transient:]
            
            train_rate = train_rate_all
            # detect from I pop
            train_cut = train_rate_I
            
            adapE = data[3][len_transient:]
            adapI = data[4][len_transient:]
            #%% UP state detection
            
            # detection parameters
#            len_state = 30 # min length of allowed state, ms 
            # shorter states are merged to previous states
            gauss_width_ratio = 10 # signal smoothing parameter
            bin_dur = 0.1 # time bin, in ms
#            ratioThreshold = 0.01 # threshold for UP state detection
            sampling_rate = 1./bin_dur
            
            # 
            dur_DOWN, dur_UP, pearsonr = UP_read.UP_duration(train_cut, 
                                                ratioThreshold = ratioThreshold,
                                                 sampling_rate = sampling_rate,
                                                 len_state = len_state, 
                                                 gauss_width_ratio = gauss_width_ratio)
            
            
            #%% exclude first or last state of simulation when shorter than len_state 
            idx_clean = np.where(np.logical_and(dur_DOWN > len_state,
                                                dur_UP > len_state))
            dur_DOWN_clean = dur_DOWN[idx_clean]
            dur_UP_clean = dur_UP[idx_clean]
            
            #%%
            plt.plot(dur_DOWN_clean/1000,dur_UP_clean/1000, '.', 
                     color = cmap(int(isim/10*255)), 
                     label = str(round(noise_range[isim], 1)))
            
            plt.xlabel('previous DOWN state duration (ms)')
            plt.ylabel('UP state duration (ms)')
            plt.title('Network simulation')
        
            dur_DOWN_all.append(dur_DOWN_clean)
            dur_UP_all.append(dur_UP_clean)
            
            dur_UP_mean[isim] = np.mean(dur_UP_clean)
            dur_UP_std[isim] = np.std(dur_UP_clean)/np.sqrt(len(dur_UP_clean))
            dur_DOWN_mean[isim] = np.mean(dur_DOWN_clean)
            dur_DOWN_std[isim] = np.std(dur_DOWN_clean)/np.sqrt(len(dur_DOWN_clean))
            
            print('PEARSON CORRELATION: ',stats.pearsonr(dur_DOWN_clean, dur_UP_clean))
            pearson_all[isim], pvalue_all[isim] = stats.pearsonr(dur_DOWN_clean, 
                       dur_UP_clean)
            
#            # shuffling UP-DOWN duration, for significance tests
#            Nshuffles = 100
#            shuffles_l = np.zeros(Nshuffles)
#            for i_sh in range(Nshuffles):
#                np.random.shuffle(dur_DOWN_clean)
#                shuffles_l[i_sh] = stats.pearsonr(dur_DOWN_clean, dur_UP_clean)[0]
#            pearson_shuffled[isim] = np.mean(shuffles_l)
#            err_shuffled[isim] = np.std(shuffles_l)/np.sqrt(Nshuffles)
#            isim +=1
            
        # statistics over all points, over all values of noise
        pearson_allparams[iilen, jjthresh], pval_allparams[iilen, jjthresh]\
        = stats.pearsonr(np.concatenate(dur_DOWN_all), 
                             np.concatenate(dur_UP_all))
        print(len_state, ratioThreshold, 'OVERALL PEARSON CORRELATION: ',
              stats.pearsonr(np.concatenate(dur_DOWN_all), 
                             np.concatenate(dur_UP_all)))
        
        #%%
        
pearson_allparams_sig = pearson_allparams
#pearson_allparams_sig[pval_allparams > 0.05] = np.nan
plt.figure()
plt.imshow(np.transpose(pearson_allparams_sig), cmap = 'RdBu_r', 
           vmin = -0.25, vmax = 0.25)
plt.colorbar()
plt.scatter(1,1, marker = 'o', color = 'k')
plt.xticks(np.arange(len(len_state_all)), len_state_all)
plt.yticks(np.arange(len(ratioThreshold_all)), ratioThreshold_all)
plt.xlabel('Minimum state length included (ms)')
plt.ylabel('Firing rate ratio threshold')
plt.savefig('paramscan_badap'+str(badap)+'.pdf')