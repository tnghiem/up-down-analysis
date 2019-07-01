# -*- coding: utf-8 -*-
"""
UP state detection from time signal.
Following the method of Renart 2010 Science. 

Used to analyse experimental and simulation data in
Nghiem T.-A.E. et al, https://www.biorxiv.org/content/10.1101/430405v3
"""
import numpy as np
from scipy import signal
from scipy import stats 
import matplotlib.pyplot as plt

bin_dur = 1
ratioThreshold = 0.4
sampling_rate = 10./50. # ms**-1
len_state = 50 # ms
#adap = 120


t_kick_cutoff = 1000


def read_rate(Nsims, adap, ratioCond = None, extInput = None):
    '''
    from file characteristics (adaptation, conductance ratio, external input)
    read population rate from file
    return population rate
    '''
    rate_exc = np.load('example_0_adap'+str(adap)\
                       +'_ratioCond'+str(ratioCond)\
                       +'_extInput'+str(extInput)\
                       +'.npy', encoding = 'latin1')[2]
    
    train_all = np.zeros((Nsims, len(rate_exc) - t_kick_cutoff))
    
    for i_file in range(Nsims):  
        filename = 'example_'+str(i_file)+'_adap'+str(adap)\
                       +'_ratioCond'+str(ratioCond)\
                       +'_extInput'+str(extInput)\
                       +'.npy'
        time_array, rate_array, rate_exc, rate_inh,\
             Raster_exc, Raster_inh, Vm_exc, Vm_inh,\
            Ge_exc, Ge_inh, Gi_exc, Gi_inh = np.load(filename, 
                                                     encoding = 'latin1')
        train_all[i_file] = rate_exc[t_kick_cutoff:] + rate_inh[t_kick_cutoff:]
        return train_all


def detect_UP(train_cut, ratioThreshold = ratioThreshold,
              sampling_rate = sampling_rate, len_state = len_state, 
              gauss_width_ratio = 10):
    '''
    detect UP states from time signal 
    (population rate or population spikes or cell voltage trace)
    return start and ends of states
    '''    
    # convolve with Gaussian
    time = range(len(train_cut))
    gauss_width = gauss_width_ratio*sampling_rate
    gauss_filter = np.exp(-0.5*((np.subtract(time,len(train_cut)/2.0)\
        /gauss_width)**2))
    gauss_norm = np.sqrt(2*np.pi*gauss_width**2)
    gauss_filter = gauss_filter/gauss_norm

    train_filtered = signal.fftconvolve(train_cut, gauss_filter)
    train_filt = train_filtered[int(len(train_cut)/2.0):\
                                int(3*len(train_cut)/2.0)]
    thresh = ratioThreshold*np.max(train_filt)
    
    # times at which filtered signal crosses threshold
    train_shift = np.subtract(train_filt, thresh)
    idx = np.where(np.multiply(train_shift[1:], train_shift[:-1]) < 0)[0]
    
    # train of 0 in DOWN state and 1 in UP state
    train_bool = np.zeros(len(train_shift))
    train_bool[train_shift > 0] = 1 # assign to 1 in UP states
    
    # cut states shorter than min length
    idx = np.concatenate(([0], idx, [len(train_filt)]))
    diff_remove = np.where(np.diff(idx) < len_state*sampling_rate)[0]
    idx_start_remove = idx[diff_remove]
    idx_end_remove = idx[np.add(diff_remove, 1)] + 1
    
    for ii_start, ii_end in zip(idx_start_remove, idx_end_remove):
        train_bool[ii_start:ii_end] = np.ones_like(\
                  train_bool[ii_start:ii_end])*train_bool[ii_start - 1] 
        # assign to same state as previous long
    
    
    idx = np.where(np.diff(train_bool) != 0)[0]
    idx = np.concatenate(([0], idx, [len(train_filt)]))/sampling_rate
    return idx, train_shift, train_bool


def UP_duration(train_cut, ratioThreshold = ratioThreshold,
                sampling_rate = sampling_rate, 
                len_state = len_state, gauss_width_ratio = 10, 
                plotting = False):
    '''
    compute UP and DOWN state durations from times of UP/DOWN transitions
    if signal starts with UP state, first UP is removed
    if signal end with DOWN state, last DOWN is removed
    for UP to previous DOWN state duration correlation analysis
    '''
    idx, train_shift, train_bool = detect_UP(train_cut, 
                                             ratioThreshold = ratioThreshold,
                                 sampling_rate = sampling_rate, 
                len_state = len_state, gauss_width_ratio = gauss_width_ratio)
    
    durs = np.diff(idx) # durations of all states
    
    if train_bool[0] == 0: # starts with DOWN state
        dur_DOWN = durs[np.arange(0, len(durs), 2)] # even states are DOWN
        dur_UP = durs[np.arange(1, len(durs), 2)] # odd states are UP
        if len(durs)%2 == 1: # ends with DOWN
            dur_DOWN_l = dur_DOWN[:-1] # discard last DOWN
        else: # ends with UP
            dur_DOWN_l = dur_DOWN
        dur_UP_l = dur_UP
               
                
    else:# starts with UP state
        dur_UP = durs[np.arange(0, len(durs), 2)] # even states are UP
        dur_DOWN = durs[np.arange(1, len(durs), 2)] # odd states are DOWN
        if len(durs)%2 == 0:# ends with DOWN    
            dur_DOWN_l = dur_DOWN[:-1] # discard last DOWN
        else:
            dur_DOWN_l = dur_DOWN
        dur_UP_l = dur_UP[1:]
    if plotting == True:
        return dur_DOWN_l, dur_UP_l, \
            stats.pearsonr(dur_DOWN_l,dur_UP_l), train_bool

    else:
        return dur_DOWN_l, dur_UP_l, stats.pearsonr(dur_DOWN_l,dur_UP_l)

def UP_rate(train_cut, ratioThreshold = ratioThreshold,
            train_rate = [0], sampling_rate = sampling_rate,
            len_state = len_state, gauss_width_ratio = 10):
    ''' UP state firing rate
    train_rate: only compute firing rate from this population, 
    but detection done on whole population
    '''
    idx, train_shift, train_bool = detect_UP(train_cut, 
                                             ratioThreshold = ratioThreshold,
                                 sampling_rate = sampling_rate, 
                len_state = len_state, gauss_width_ratio = gauss_width_ratio)
    if len(train_rate) == 1:
        train_rate = train_cut
    
    # idx of state starts and ends
    if train_bool[0] == 0: # starts with DOWN state
        idx_start_up = idx[np.arange(1,len(idx),2)]
        idx_end_up = idx[np.arange(2,len(idx),2)]
        if len(idx)%2 == 0: # ends with DOWN
            idx_start_up = idx_start_up[:-1]#np.concatenate((idx_end_up,[len(train_shift)]))

    else:# starts with UP state
        idx_start_up = idx[np.arange(2,len(idx),2)] 
        idx_end_up = idx[np.arange(3,len(idx),2)] # discard first UP as no previous DOWN
        if len(idx)%2 == 1: # ends with DOWN
            idx_start_up = idx_start_up[:-1]#np.concatenate((idx_end_up,[len(train_shift)]))    

    # frequency of UP state
    pA_up = np.zeros(len(idx_end_up))
    for ii in range(len(idx_end_up)):
        pA_up[ii] = np.mean(train_rate[int(idx_start_up[ii]*sampling_rate):\
                            int(idx_end_up[ii]*sampling_rate)])
#        pA_up_all.append(pA_up)
    return pA_up

def UP_onset(train_cut, ratioThreshold = ratioThreshold,
            train_rate = [0], sampling_rate = sampling_rate,
            len_state = len_state, gauss_width_ratio = 10, 
            dur_onset = 50.):
    ''' UP state firing rate
    train_rate: only compute firing rate from this population, 
    but detection done on whole population
    rate computed in 
    '''
    idx, train_shift, train_bool = detect_UP(train_cut, ratioThreshold = ratioThreshold,
                                 sampling_rate = sampling_rate, 
                len_state = len_state, gauss_width_ratio = gauss_width_ratio)
    if len(train_rate) == 1:
        train_rate = train_cut
    
    # idx of state starts and ends
    if train_bool[0] == 0: # starts with DOWN state
        idx_start_up = idx[np.arange(1,len(idx),2)]
        idx_end_up = idx[np.arange(2,len(idx),2)]
        if len(idx)%2 == 0: # ends with DOWN
            idx_start_up = idx_start_up[:-1]#np.concatenate((idx_end_up,[len(train_shift)]))

    else:# starts with UP state
        idx_start_up = idx[np.arange(2,len(idx),2)] 
        idx_end_up = idx[np.arange(3,len(idx),2)] # discard first UP as no previous DOWN
        if len(idx)%2 == 1: # ends with DOWN
            idx_start_up = idx_start_up[:-1]#np.concatenate((idx_end_up,[len(train_shift)]))    

    # frequency of UP state
    pA_up = np.zeros(len(idx_end_up))
    for ii in range(len(idx_end_up)):
        train_whole = train_rate[int(idx_start_up[ii]*sampling_rate):\
                            int(idx_end_up[ii]*sampling_rate)]
        pA_up[ii] = np.mean(train_whole[:int(dur_onset*sampling_rate)])
#        pA_up_all.append(pA_up)
    return pA_up

	
def plot_UP_onset(trace, ratioThreshold = ratioThreshold, dur_DOWN_min = 0,
                  dur_DOWN_max = 1e5, len_window = 500, 
                  sampling_rate = sampling_rate, len_state = len_state):
    '''
    plot UP states aligned to onset
    after DOWN state in range of durations
    trace after cutting first 5000 ms i.e. 1000 time bins
    '''
    plt.figure()
    # cut the first 5000 ms of the trace!
    dur_DOWN, dur_UP = UP_duration(trace, ratioThreshold = ratioThreshold,
                                   sampling_rate = sampling_rate, 
                                   len_state = len_state)[:2]
    times = np.arange(0, 1.*len(trace)/sampling_rate, 
                      1./sampling_rate) # time in ms
#    plt.plot(times, trace, 'b')

    for i_state in range(len(dur_UP)):
        if dur_DOWN[i_state] > dur_DOWN_min and \
            dur_DOWN[i_state] < dur_DOWN_max:
                t_onset = np.sum(dur_DOWN[:i_state + 1]) + \
                    np.sum(dur_UP[:i_state]) # onset of UP after selected DOWN

                t_start = np.max((0, t_onset - len_window/10.))
                t_end = np.min((t_onset + 9*len_window/10., t_onset + \
                                dur_UP[i_state])) # till end of following UP
                
                idx_select = np.where(np.logical_and(times >= t_start, 
                                                       times < t_end))

                plt.plot(np.arange(t_start, t_end, 1./sampling_rate) - t_start, 
                         trace[idx_select])
    plt.xlim(0, len_window)
    plt.ylim(0, 0.8)
    plt.title('DOWN duration between '+str(dur_DOWN_min)+ \
               ' and '+str(dur_DOWN_max)+' ms')
    plt.show()
    return pA_up


def plot_UP_onset_mean(trace, trace_rate = [0], ratioThreshold = ratioThreshold,
                       dur_DOWN_min = 0,
                  dur_DOWN_max = 1e5, len_window = 500, 
                  sampling_rate = sampling_rate, len_state = len_state, 
                  gauss_width_ratio = 10, color_curr = 'k' ):
    '''
    plot UP states aligned to onset
    after DOWN state in range of durations
    trace after cutting first 5000 ms i.e. 1000 time bins
    '''
    if len(trace_rate) ==1:
        trace_rate = trace
    plt.figure()
    # cut the first 5000 ms of the trace!
    dur_DOWN, dur_UP = UP_duration(trace, ratioThreshold = ratioThreshold,
                                   sampling_rate = sampling_rate, 
                                   len_state = len_state,
                                   gauss_width_ratio = 10, )[:2]
    times = np.arange(0, 1.*len(trace)/sampling_rate, 
                      1./sampling_rate) # time in ms
#    plt.plot(times, trace, 'b')
    trace = trace_rate
    curves_all = []
    for i_state in range(len(dur_UP)):
        if dur_DOWN[i_state] > dur_DOWN_min and \
            dur_DOWN[i_state] < dur_DOWN_max:
                t_onset = np.sum(dur_DOWN[:i_state + 1]) + \
                    np.sum(dur_UP[:i_state]) # onset of UP after selected DOWN

                t_start = np.max((0, t_onset - len_window/10.))
                t_end = np.min((t_onset + 9*len_window/10., t_onset + \
                                dur_UP[i_state])) # till end of following UP
                
                idx_select = np.where(np.logical_and(times >= t_start, 
                                                       times < t_end))

                curve_curr = np.zeros(len_window)
                curve_curr[:int((t_end - t_start)*sampling_rate)] =\
                    trace[idx_select]
                curves_all.append(curve_curr)
    curve_mean = np.mean(np.array(curves_all), axis = 0)
    curve_std = np.std(np.array(curves_all), axis = 0)/np.sqrt(len(dur_UP))
    
    plt.plot(np.arange(0, len_window, 1./sampling_rate), 
             curve_mean, color_curr)
    plt.fill_between(np.arange(0, len_window, 1./sampling_rate), 
                     curve_mean - curve_std, 
                     curve_mean + curve_std, 
                     color = color_curr, alpha = 0.5)
                
    plt.xlim(0, len_window)
    plt.ylim(-0.05, 0.2)
    plt.title('DOWN duration between '+str(dur_DOWN_min)+ \
               ' and '+str(dur_DOWN_max)+' ms')
    plt.ylim(0, 0.5)

    plt.show()
    return np.arange(0, len_window, 1./sampling_rate), curve_mean, curve_std

def plot_UP_onset_mean_sameEI(trace, trace_rate_E, trace_rate_I,
                ratioThreshold = ratioThreshold, dur_DOWN_min = 0,
                  dur_DOWN_max = 1e5, len_window = 500, 
                  sampling_rate = sampling_rate, len_state = len_state, 
                  gauss_width_ratio = 0 ):
    '''
    plot UP states aligned to onset
    after DOWN state in range of durations
    trace after cutting first 5000 ms i.e. 1000 time bins
    '''

    plt.figure()
    # cut the first 5000 ms of the trace!
    dur_DOWN, dur_UP = UP_duration(trace, ratioThreshold = ratioThreshold,
                                   sampling_rate = sampling_rate, 
                                   len_state = len_state,
                                   gauss_width_ratio = 10 )[:2]
    times = np.arange(0, 1.*len(trace)/sampling_rate, 
                      1./sampling_rate) # time in ms

    col_cycle = ['b','r']
    label_cycle = ['E','I']
    i_col = 0
    for trace in [trace_rate_E, trace_rate_I]:
        curves_all = []
        for i_state in range(len(dur_UP)):
            if dur_DOWN[i_state] > dur_DOWN_min and \
                dur_DOWN[i_state] < dur_DOWN_max:
                    t_onset = np.sum(dur_DOWN[:i_state + 1]) + \
                        np.sum(dur_UP[:i_state]) # onset of UP after selected DOWN

                    t_start = np.max((0, t_onset - len_window/10.))
                    t_end = np.min((t_onset + 9*len_window/10., t_onset + \
                                    dur_UP[i_state])) # till end of following UP
                    
                    idx_select = np.where(np.logical_and(times >= t_start, 
                                                           times < t_end))

                    curve_curr = np.zeros(len_window)
                    curve_curr[:int((t_end - t_start)*sampling_rate)] =\
                        trace[idx_select]
                    curves_all.append(curve_curr) 
        curve_mean = np.mean(np.array(curves_all), axis = 0)
        curve_std = np.std(np.array(curves_all), axis = 0)/np.sqrt(len(dur_UP))
        
        plt.plot(np.arange(0, len_window, 1./sampling_rate), 
                 curve_mean, color = col_cycle[i_col], 
                 label = label_cycle[i_col])
        plt.fill_between(np.arange(0, len_window, 1./sampling_rate), 
                         curve_mean - curve_std, 
                         curve_mean + curve_std, 
                         color = col_cycle[i_col], alpha = 0.5)
        i_col += 1
                
    plt.xlim(0, len_window)

    plt.legend()
    plt.title('DOWN duration between '+str(dur_DOWN_min)+ \
               ' and '+str(dur_DOWN_max)+' ms')
    plt.savefig('alignedUP_'+str(dur_DOWN_min)+'_'+\
                str(dur_DOWN_min)+'_'+str(ratioThreshold), format='pdf')    

    plt.show()
    return np.arange(0, len_window, 1./sampling_rate), curve_mean, curve_std

def plot_UP_onset_mean_separate(trace, trace_rate_E, trace_rate_I,
                ratioThreshold = ratioThreshold, dur_DOWN_min = 0,
                  dur_DOWN_max = 1e5, len_window = 500, 
                  sampling_rate = sampling_rate, len_state = len_state, 
                  gauss_width_ratio = 0 ):
    '''
    plot UP states aligned to onset
    after DOWN state in range of durations
    trace after cutting first 5000 ms i.e. 1000 time bins
    '''

    plt.figure()
    # cut the first 5000 ms of the trace!
    dur_DOWN, dur_UP = UP_duration(trace, ratioThreshold = ratioThreshold,
                                   sampling_rate = sampling_rate, 
                                   len_state = len_state,
                                   gauss_width_ratio = 10 )[:2]
    times = np.arange(0, 1.*len(trace)/sampling_rate, 
                      1./sampling_rate) # time in ms

    col_cycle = ['b','r']
    label_cycle = ['E','I']
    curves_alltypes = {}
    for i_state in range(len(dur_UP)):
        curves_all = []
        if dur_DOWN[i_state] > dur_DOWN_min and \
                dur_DOWN[i_state] < dur_DOWN_max:
            plt.figure()
            i_col = 0
            for trace in [trace_rate_E, trace_rate_I]:
                t_onset = np.sum(dur_DOWN[:i_state + 1]) + \
                    np.sum(dur_UP[:i_state]) # onset of UP after selected DOWN

                t_start = np.max((0, t_onset - len_window/10.))
                t_end = np.min((t_onset + 9*len_window/10., t_onset + \
                                dur_UP[i_state])) # till end of following UP
                
                idx_select = np.where(np.logical_and(times >= t_start, 
                                                       times < t_end))
                curve_curr = np.zeros(len_window)
                curve_curr[:int((t_end - t_start)*sampling_rate)] =\
                    trace[idx_select]
                curves_all.append(curve_curr) 
                plt.plot(np.arange(0, len_window, 1./sampling_rate), 
                 curve_curr, color = col_cycle[i_col], 
                 label = label_cycle[i_col])
                i_col += 1
            plt.ylim(0, 0.3)
            plt.legend()
                
    plt.xlim(0, len_window)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.title('DOWN duration between '+str(dur_DOWN_min)+ \
               ' and '+str(dur_DOWN_max)+' ms')
    plt.savefig('alignedUP_'+str(dur_DOWN_min)+'_'+\
                str(dur_DOWN_min)+'_'+str(ratioThreshold), format='pdf')    

    plt.show()
    return curves_alltypes
