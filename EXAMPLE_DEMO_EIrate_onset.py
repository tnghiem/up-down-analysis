#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse UP and DOWN state duration from simulated data
Effect of excitatory (E) and inhibitory (I) firing rates on UP state duration

Used to produce fig. 2B,C of paper: 
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

#E and I rates in all UP states
rateE_UP_all = []
rateI_UP_all = []

noise_all = []

# statistics of UP-DOWN state durations
dur_UP_mean = np.zeros(Nsims)
dur_UP_std = np.zeros(Nsims)
dur_DOWN_mean = np.zeros(Nsims)
dur_DOWN_std = np.zeros(Nsims)

rateE_UP_mean = np.zeros(Nsims)
rateI_UP_mean = np.zeros(Nsims)
rateE_UP_std = np.zeros(Nsims)
rateI_UP_std = np.zeros(Nsims)

rateEI_ratio_mean = np.zeros(Nsims)
rateEI_ratio_std = np.zeros(Nsims)

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
                                         gauss_width_ratio = gauss_width_ratio)

#%% firing rate in each UP state    
    rateE_UP = UP_read.UP_onset(train_cut, ratioThreshold = ratioThreshold,
            train_rate = train_rate_E, sampling_rate = sampling_rate,
            len_state = len_state, gauss_width_ratio = 10)

    rateI_UP = UP_read.UP_onset(train_cut, ratioThreshold = ratioThreshold,
            train_rate = train_rate_I, sampling_rate = sampling_rate,
            len_state = len_state, gauss_width_ratio = 10) 
    
    rateE_UP_all.append(rateE_UP)
    rateI_UP_all.append(rateI_UP)
    noise_all.append(np.ones_like(rateI_UP)*noise_range[isim])
    
    rate_EI_ratio = rateE_UP/rateI_UP
    rateEI_ratio_mean[isim] = np.mean(rate_EI_ratio)
    rateEI_ratio_std[isim] = np.std(rate_EI_ratio)
    
    rateE_UP_mean[isim] = np.mean(rateE_UP)
    rateI_UP_mean[isim] = np.mean(rateI_UP)
    rateE_UP_std[isim] = np.std(rateE_UP)
    rateI_UP_std[isim] = np.std(rateI_UP)
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

#%% plot mean E and I firing rates at UP state onset
plt.figure()
plt.plot(noise_range, rateE_UP_mean, 'b', label = 'Excitatory')
plt.plot(noise_range, rateI_UP_mean, 'r', label = 'Inhibitory')
plt.fill_between(noise_range, rateE_UP_mean - rateE_UP_std,
                 rateE_UP_mean + rateE_UP_std, 
                 color = 'b', alpha = 0.5)
plt.fill_between(noise_range, rateI_UP_mean - rateI_UP_std,
                 rateI_UP_mean + rateI_UP_std, 
                 color = 'r', alpha = 0.5)
plt.xlabel('External noise amplitude (pA)')
plt.ylabel('Firing rate at UP state onset (Hz)')
plt.legend()

#%% E/I rate ratio vs noise amplitude

plt.figure()
noise_l = np.concatenate(noise_all)
ratio_l = np.concatenate(rateE_UP_all)/np.concatenate(rateI_UP_all)
means = np.zeros(Nsims)
stderrs = np.zeros(Nsims)

for isim in range(Nsims):
    means[isim] = np.mean(rateE_UP_all[isim]/rateI_UP_all[isim])
    stderrs[isim] = np.std(rateE_UP_all[isim]/rateI_UP_all[isim])/np.sqrt(len(rateE_UP_all[isim]))
#    plt.plot(noise_jitter_all[isim], rateE_UP_all[isim]/rateI_UP_all[isim],'.',
#             color = cmap(int(isim/10*255)))
#plt.plot(noise_jitter_l, ratio_l, 'k.')
#plt.plot(noise_l, means, yerr = stderrs)
#plt.plot(noise_range, means, 'o')
plt.errorbar(noise_range, means, yerr = stderrs, fmt = 'o', color = '0.5')
#plt.fill_between(noise_range, means - stderrs, means + stderrs, alpha = 0.5)
a,b = np.polyfit(noise_range, means, 1)
plt.plot(noise_range, noise_range*a + b, 'k')
plt.xlabel('External noise amplitude (pA)')
plt.ylabel('E/I rate ratio')
plt.savefig('EI_ratio_errorbar_noise_amp.pdf')
#plt.ylim(0.65, 0.75)

print(stats.pearsonr(noise_l, ratio_l))

#%% E/I rate ratio histograms
plt.figure()
for isim in range(Nsims):
    counts, bins = np.histogram(rateE_UP_all[isim]/rateI_UP_all[isim], 
             bins = np.linspace(0.55, 0.85,15))
    plt.plot(bins[:-1], counts/np.sum(counts)*100, 
             color = cmap(int(isim/10*255)))
plt.xlabel('E/I rate ratio')
plt.ylabel('% of UP states')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#%% all E/I rate ratios
ratio_clean_l = []
for isim in range(Nsims):
    ratio_curr = rateE_UP_all[isim]/rateI_UP_all[isim]
    dur_UP_curr = dur_UP_all[isim]
    if len(ratio_curr) == len(dur_UP_curr): # first state is DOWN   
        ratio_clean_l.append(ratio_curr)
    else: # first state is UP, to discard
        ratio_clean_l.append(ratio_curr[1:])

dur_UP_l = np.concatenate(dur_UP_all)
ratio_clean_l = np.concatenate(ratio_clean_l)

#%% binning UP states into bins of E/I ratio at onset
nbins = 6
ratio_bins = np.linspace(0.62,0.8,nbins)
dur_UP_per_ratio_mean = np.zeros(nbins - 1)
dur_UP_per_ratio_max = np.zeros(nbins - 1)
dur_UP_per_ratio_min = np.zeros(nbins - 1)
dur_UP_per_ratio_std = np.zeros(nbins - 1)

dur_UP_idx_all = []
for i_ratio in range(len(ratio_bins) -1):
    idx = np.where(np.logical_and(ratio_clean_l > ratio_bins[i_ratio],
                               ratio_clean_l < ratio_bins[i_ratio + 1]))
    dur_UP_idx = dur_UP_l[idx]
    if len(dur_UP_idx > 0):
        dur_UP_idx_all.append(dur_UP_idx)
    
#%%  UP state duration histogram plotting
fig, axs = plt.subplots(nrows = nbins-1)
cmap = plt.get_cmap('coolwarm_r')
for i_bin in range(nbins-1):
    axs[i_bin].hist(dur_UP_idx_all[i_bin], bins = np.linspace(0,2500,20),
       density = True, 
       color = cmap(int(i_bin/(nbins - 1)*255)),
       label = str('%.2f'%ratio_bins[i_bin]) \
       + ' < E/I ratio < '+str('%.2f'%ratio_bins[i_bin + 1]))
    axs[i_bin].set_yscale('log')
    axs[i_bin].legend(frameon=False, bbox_to_anchor=(1.05, 1), 
       loc=2, borderaxespad=0.)
plt.xlabel('UP state duration (s)')
plt.savefig('EIratio_UPdur_bins_hist.pdf')