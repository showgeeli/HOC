# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:30:53 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""


from src import Correlated_Poisson_Process as pp
from src import Plot as plot
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# generate parallel spikes with different rates
spike_trains = pp.generate_parallelST([0.2,0.2,0.4,0.4,0.4,0.2,0.1], 10) 
marked_process1 = pp.parallelST_to_markedProcess(spike_trains)
# reverse marked process to parallel spike trains. 
components = pp.markedProcess_to_components(marked_process1)

correlatedST1 = pp.markedProcess_to_correlatedST(marked_process1)

# generate marked process with list of rates
marked_process2 = pp.generate_markedProcess([0.2,0.2,0.2,0.4,0.4,0.4,0.8], 10)
correlatedST2 = pp.markedProcess_to_correlatedST(marked_process2)

# use rate distribution to generate marked process directly
marked_process3 = pp.direct_markedProcess([0.2,0.2,0.2,0.4,0.4,0.4,0.5], 10)
correlatedST3 = pp.markedProcess_to_correlatedST(marked_process3)


# plot corrleated poisson process. Same amplitude has same color
plot.colorPlot_correlatedST(marked_process3, color_key='amplitudes')
# plot corrleated poisson process. Spikes coming from same component have same color.
# Color map can be changed, default is plt.cm.hsv
plot.colorPlot_correlatedST(marked_process3, color_key='components',color_map = plt.cm.tab20)


# stimulate a Poisson process and thin it.
rate = 40       # rate (λ) of the target spike train
cv = 0.2        # coefficiant of variation (cv) of the target spike train
T = 500 
sigma = np.sqrt(np.log(cv**2 +1))             # analytical expression of σ
mu = -0.5 * np.log(rate**2 * (cv**2 +1))      # analytical expression of μ

def hazard(t):
    pdf = lognorm.pdf(t, sigma, np.exp(mu)) # probability density function
    sf = lognorm.sf(t, sigma, np.exp(mu))   # survival function
    return pdf / sf

# define R_m, which is a bit greater than peak value of hazard function.
# With bigger R_m more spikes will be rejected. With smaller R_m, more 
# spikes will be kept, which makes an efficient thinning process.
x = np.linspace(0, T, 300)
R_m = max(hazard(x)) * 1.1 # magnification factor 1.1. It can be replaced.

S = pp.generate_parallelST([R_m], T)[0] # Poisson process with rate R_m
new_S = pp.thinning(S, rate, cv, T)
# print the number of spikes before and after thinning
print(len(S))
print(len(new_S))
isi_logn = np.diff(new_S)
plt.figure()
count, bins, ignored = plt.hist(isi_logn,30, density=True, align='mid')
xs = np.linspace(min(bins),max(bins), 300)
pdf =lognorm.pdf(xs, sigma, np.exp(mu))
plt.plot(xs, pdf, c = 'k')

'''For decompound'''
# alternative to create population spike trains
amplitudes_dist = [0.6, 0, 0, 0, 0, 0, 0, 0, 0.4, 0]
correlatedST = pp.generate_popuActivity(amplitudes_dist, 30, T)
#rate_of_amplitudes = np.array(amplitudes_dist) * h
# caculate counts per spike trains 
counts= [np.array([])]*nn
bin_edges2= [np.array([])]*nn
for i in range(nn):
    count, bin_edge= np.histogram(correlatedST[i], bins = L, range =(0, T), density= False)
    counts[i]=count
    bin_edges2[i] = bin_edge   
counts = np.sum(counts,axis = 0)
