# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 01:39:24 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""
from src import Poisson_Process as pp
import time

# initialize class instance
st = pp.Stationary_Poisson_Process_ST()

# create 7 independent poisson processes as components
spike_trains = st.generate(0.2, 10,7)
# plot the components
pp.colorful_plot(spike_trains)
# create 3 correlated poisson spike trains
neuron_spikes, color, amplitude = st.correlate(spike_trains)
# plot the 3 correlated poisson spike trains
pp.correlation_plot(neuron_spikes, color)

print(amplitude)



starttime = time.process_time() 
# generate correalated poisson process for arbitary number of neurons
# lambda0 = 0.2, T = 10, number of neurons = 4
# As tried, 7 neurons is fine. 10 neurons took several minutes to generate the graph.
correlated_spikes, event_code = st.correlation_generate(0.2, 10, 7) 

# plot the correlated spike trains
pp.correlation_plot(correlated_spikes, event_code)

print(time.process_time()- starttime, 'second') # measure cpu time


import numpy as np
spike_trains.append(np.array([]))
spike_trains.append(np.array([6.5435345]))
np.shape(spike_trains)[0]
len(spike_trains)
n = np.size(spike_trains)
nt = spike_trains.count((spike_trains[i].size > 0 for i in range(n)))
print(list(filter(None,spike_trains)))
print( list(filter(None, [1, 45, "", 6, 50, 0, {}, False])) ) 
count = sum(1 for e in spike_trains if e is not None)
