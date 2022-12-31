# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:08:40 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""

from src import Correlated_Poisson_Process as pp
from src import Plot as plot

# generate parallel spikes
spike_trains = pp.generate_parallelST(0.2, 10,7)
# plot without color
plot.plain_plot(spike_trains)
# generate marked process with the given parallel spike trains for abitrary number of neurons
marked_process = pp.parallelST_to_markedProcess(spike_trains, 3)


# plot corrleated poisson process. Same amplitude has same color
plot.correlated_colorPlot(marked_process, color_key='amplitudes')
# plot corrleated poisson process. Spikes coming from same component have same color.
plot.correlated_colorPlot(marked_process, color_key='components')


# generate marked process, with rates, T, abitrary number of neurons as input parameters.
marked_process2 = pp.generate_markedProcess([0.2,0.2, 0.3, 0.3,0.4,0.4,0.4], 10, 3)
# plot in color according to the number of components
plot.correlated_colorPlot(marked_process2, color_key='amplitudes')
