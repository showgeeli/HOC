# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:53:26 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""
from src import Poisson_Process as pp
# create an instance  of class "Stationary_Poisson_Process_ST"
st = pp.Stationary_Poisson_Process_ST()

# generate spike trains
# lambda0 = 0.5; T = 10; number of spikes = 
spike_trains = st.generate(0.5, 10,7)
print(spike_trains)
# plot the generated spike trains, with different color for each spike train
pp.colorful_plot(spike_trains)

# Compound the first n-1 spike trains with the n_th spike trains. The last spike train is the commen input.
compounded_spike_trains= st.compound(spike_trains)
print(compounded_spike_trains)

# plot the compounded spike trains, with common input marked as red.
pp.compound_plot(compounded_spike_trains)




