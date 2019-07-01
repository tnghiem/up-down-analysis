# up-down-analysis
Code from the paper: Nghiem T.-A. E. et al, Cholinergic switch between two different types of slow waves in cerebral cortex https://www.biorxiv.org/content/10.1101/430405v3

Scripts used to analyse traces from simulated as well as experimentally recorded neural activity, presenting UP and DOWN state dynamics. Can be run on the simulation data also included here. 

# contents
data_noise3_adap10: simulation data with low adaptation strength b = 10 nS, produces 'sleep-like' dynamics, used for Fig. 2-3

data_noise3_adap50: simulation data with high adaptation strength b = 50 nS, produces 'anesthesia-like' dynamics, used for Fig. 2-3

UP_read.py: module of functions used to detect UP and DOWN states from signals, and compute durations and firing rates of UP and DOWN states

EXAMPLE_DEMO_UpDownDur.py: script to analyse UP and DOWN state durations and produce Fig. 2D and 3A

EXAMPLE_DEMO_EIrate_onset.py: script to analyse excitatory and inhibitory firing rates and produce Fig. 2B,C

EXAMPLE_DEMO_loop_thresh: script to analyse UP and DOWN state durations with different detection parameters and produce Fig. S3


# contact
trang-anh.nghiem@cantab.net
