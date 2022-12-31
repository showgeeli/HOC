# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:50:24 2021

@author: xiuzhi.li@merkur.uni-freiburg.de

This module is to provide a thinning process for a spike train. The thinning 
process is a renewal process with harzard function of lognormal process, see 
reference [1]. After thinning, inter-spike-interval distribution of each 
neuron should follow that lognormal distribution.

* hazard
* thinning


References
----------
.. [1] Reimer IC, Staude B, Ehm W, Rotter S. Modeling and analyzing 
   higher-order correlations in non-Poissonian spike trains. J Neurosci 
   Methods. 2012 Jun 30;208(1):18-33. doi: 10.1016/j.jneumeth.2012.04.015.
"""

from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np
from src import Correlated_Poisson_Process as pp

# define hazard function of the renewal process

    
def thinning(S, rate, cv, T):


    # Target spike train is a lognormal process. 
    # Caculate σ and μ according to rate and cv.
    sigma = np.sqrt(np.log(cv**2 +1))               # analytical expression of σ
    mu = abs(-0.5 * np.log(rate**2 * (cv**2 +1)))   # analytical expression of μ
    #print(sigma)
    #print(mu)    
    def hazard(t):
        pdf = lognorm.pdf(t, sigma, np.exp(mu))
        sf = lognorm.sf(t, sigma, np.exp(mu))
        return pdf / sf
    
    # define R_m, which is a bit greater than the peak value of hazard function.   
    x = np.linspace(0, T, 300)
    R_m = max(hazard(x)) * 1.5
    #print('R_m: ', R_m)

    #def renewal(S, hazard, R_m):
    n = len(S)
    U = np.random.uniform(0,R_m, n) # draw n random values between [0, R_m)
    t_latest = S[0] 
    new_S = [t_latest]
    for i in range(1,n):
        if U[i] <= hazard([S[i]-t_latest]):
            t_latest = S[i]
            new_S.append(t_latest)
            
    # plot the hazard function and spikes 
    # plt.figure()
    # for i in range(len(new_S)-1):
    #     x = np.linspace(new_S[i], new_S[i+1], 100)
    #     plt.plot(x, hazard(x-new_S[i]), c = 'k')
    # # plot the last part of harzard function, where it equals 0
    # x_tail = np.linspace(new_S[-1], S[-1], 50)
    # plt.plot(x_tail, hazard(x_tail-new_S[-1]), c = 'k')
    # # plot the spikes
    # y= np.empty(len(S))
    # y.fill(-0.1) 
    # y_new = np.empty(len(new_S))
    # y_new.fill(-0.1) 
    # plt.plot(S, y, marker = '|', linestyle = '', markersize=30, c = 'gray')
    # plt.plot(new_S, y_new, marker = '|', linestyle = '', markersize=30, c = 'r')
    # plt.ylim(-0.1, R_m+0.1)        
                
            
    return new_S  
 
rate = 100      # rate (λ) of the target spike train
cv = 0.3        # coefficiant variation (cv) of the target spike train
T = 500 
sigma = np.sqrt(np.log(cv**2 +1))               # analytical expression of σ
mu = abs(-0.5 * np.log(rate**2 * (cv**2 +1)))   # analytical expression of μ
S = pp.generate_ST(100, T)[0]
new_S = thinning(S, rate, cv, T)




isi_logn = np.diff(new_S)
print(len(S))        # number of spikes in poisson stimulation
print(len(isi_logn)) # number of spikes after thinning

# plot the histogramm of ISI and the pdf curve 
plt.figure()
count, bins, ignored = plt.hist(isi_logn,30, density=True, align='mid')
xs = np.linspace(min(bins),max(bins), 300)
pdf =lognorm.pdf(xs, sigma, np.exp(mu))
plt.plot(xs, pdf, c = 'k')
# pdf2 = np.exp(-np.power((np.log(xs)-mu),2) / (sigma**2 * 2)) / (xs * sigma * np.sqrt(2 * np.pi))                            
# plt.plot(xs, pdf2, label ='pdf', c = 'r')

s2, loc2, scale2 = lognorm.fit(isi_logn, floc=0)
print(s2, loc2, scale2)

## Poisson distribution as an trail
st = pp.generate_parallelST(50, 10)[0]
isi = np.diff(st)
count, bins, ignored = plt.hist(isi, bins =50, density=True, histtype = 'stepfilled', color = 'b')
l = 50 
xt = np.linspace(min(bins), max(bins), 500)
f = l * np.exp(- l * xt)
plt.plot(xt, f, label ='pdf', c = 'r')