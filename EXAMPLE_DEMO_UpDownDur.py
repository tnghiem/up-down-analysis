#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse UP and DOWN state duration from simulated data
Used to generate Fig. 2D and 3A of paper:
Nghiem T.-A.E. et al, https://www.biorxiv.org/content/10.1101/430405v3

trang-anh.nghiem@cantab.net
"""
#%% Make spike trains, select clean cells

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import UP_read


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
badap = 50 # mean adaptation strength
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
    len_state = 50 # min length of allowed state, ms 
    # shorter states are merged to previous states
    gauss_width_ratio = 10 # signal smoothing parameter
    bin_dur = 0.1 # time bin, in ms
    ratioThreshold = 0.02 # threshold for UP state detection
    sampling_rate = 1./bin_dur
    
    # 
    dur_DOWN, dur_UP, pearsonr = UP_read.UP_duration(train_cut, 
                                        ratioThreshold = ratioThreshold,
                                         sampling_rate = sampling_rate,
                                         len_state = len_state, 
                                         gauss_width_ratio = gauss_width_ratio)[:3]
    
    
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
    
# statistics over all points, over all values of noise
print('OVERALL PEARSON CORRELATION: ',
      stats.pearsonr(np.concatenate(dur_DOWN_all), np.concatenate(dur_UP_all)))

#%%plotting
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('dur_DOWN_UP_scatter_'+str(badap)+'.pdf')


# mean durations
plt.figure()
plt.plot(noise_range, dur_UP_mean, 'g')
plt.plot(noise_range, dur_DOWN_mean, '0.5')
plt.fill_between(noise_range, dur_UP_mean - dur_UP_std,
         dur_UP_mean + dur_UP_std, facecolor = 'g', alpha = 0.4, 
         label = 'UP duration')
plt.fill_between(noise_range, dur_DOWN_mean - dur_DOWN_std,
         dur_DOWN_mean + dur_DOWN_std, facecolor = 'k', alpha = 0.4, 
         label = 'DOWN duration')
plt.xlabel('External noise amplitude (pA)')
plt.ylabel('Mean state duration (s)')