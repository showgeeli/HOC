# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:35:22 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import gamma
from scipy.stats import lognorm
from math import exp
from sympy import symbols, Eq, solve
from src import Correlated_Poisson_Process as pp
rate = 0.3
x1 = np.linspace(poisson.ppf(0.01, rate), poisson.ppf(0.999, rate),100)
y1 = poisson.pmf(x1, rate)
plt.scatter(x1, y1)
r = poisson.rvs(0.9, size=100)
a=6
scale = 1
x2 = np.linspace(gamma.ppf(0.001, a, scale=scale), gamma.ppf(0.999, a,scale=scale), 100)
y2 = gamma.pdf(x2, a, scale=scale)
plt.plot(x2, y2)

sigma = 0.91
x3 = np.linspace(lognorm.ppf(0.01, sigma), lognorm.ppf(0.99, sigma), 100)
y3 = lognorm.pdf(x3, sigma)
plt.plot(x3, y3)

sf = lognorm.sf(np.linspace(0,10,100),sigma)
r = lognorm.rvs(sigma, size=100)

rate = 0.5
e = 1/rate

t,s = symbols('t, s')
a = Eq(t + s**2/2,np.log(e))
b = Eq(exp(s**2 -2*t) - exp(-2*t), 1)
s = solve([a,b],s)


# Poisson process with dead time
# pdf = λe^(-λt)
# hazard y = 0.9, E(X) = 0.9, Var(X) = 0, λ = 1/0.9, CV = 0?

rate = 1/0.9
R_m = 0.9   # let R_m = max(hazard) 
T = 10
def hazard(age):
    return [R_m if a >= 0.5 else 0  for a in age]

# S = pp.generate_parallelST(rate, T)[0]
marked_processes= pp.generate_markedProcess([0.2,0.2,0.3,0.3,0.4,0.4,0.1], 10)
correlated_spike_trains = pp.markedProcess_to_correlatedST(marked_processes)
S = correlated_spike_trains[0] # S is one spike train from 3 correlated neurons
n = len(S)
Us = [] # to store all the random U values, just for validation
Hs = [] # to store all the returned values from hazard function, just for validation
t_latest = S[0] 
new_S = [t_latest]
for i in range(1,n):
    U = np.random.uniform(0,R_m)
    Us.append(U)
    Hs.append(S[i]-t_latest)
    if U <= hazard([S[i]-t_latest])[0]:
        t_latest = S[i]
        new_S.append(t_latest)
        
plt.figure()
# plot the hazard function
for i in range(len(new_S)-1):
    x = np.linspace(new_S[i], new_S[i+1], 100)
    plt.plot(x, hazard(x-new_S[i]), c = 'k')
# plot the last part of harzard function, where it equals 0
x_tail = np.linspace(new_S[-1], S[-1], 50)
plt.plot(x_tail, hazard(x_tail-new_S[-1]), c = 'k')
# plot the spikes
y= np.empty(len(S))
y.fill(-0.1) 
y_new = np.empty(len(new_S))
y_new.fill(-0.1) 
plt.plot(S, y, marker = '|', linestyle = '', markersize=30, c = 'gray')
plt.plot(new_S, y_new, marker = '|', linestyle = '', markersize=30, c = 'r')
plt.ylim(-0.1, R_m+0.1)
   