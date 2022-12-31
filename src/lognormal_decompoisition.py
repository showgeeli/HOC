# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:20:15 2021

@author: Willkommen
"""


import matplotlib.pyplot as plt
import numpy as np 
from src import Correlated_Poisson_Process as pp
from scipy.stats import lognorm
# find index of specific amlitude in the rate list 
def index_of_amplitude(nn, amplitude):
    def to_binary(number,n):
        return ('{0:0'+str(n)+'b}').format(number)
    
    ampl = []
    
    for i in range(1, 2**nn):
        binary = to_binary(i,nn)
        ampl.append(binary.count("1"))
    ampl = np.array(ampl)
    return np.where(ampl==amplitude)

# convert an integer into binary code and count how many ones are there.
# Input parameter is a list of integers.
def count_ones(n):
    ones = np.zeros(len(n), dtype = int)
    for i in range(len(n)):
        ones[i] = bin(n[i]).count("1")  
    return ones


T = 100       # observation duration
sample_size =100     # smaple size of theta
thetas = np.arange(- np.pi, np.pi, step = 2 * np.pi / sample_size)
nn = 10       # analytical number of neurons
L = 10000
       # nubmer of bin
h = T/L      # bin size
n_v = 10

index_ampl1 = index_of_amplitude(nn, 1)
index_ampl9 = index_of_amplitude(nn, 9)
rate = np.zeros(2 ** nn -1)
rate[index_ampl1]= 0.9 
rate[index_ampl9]= 1.4

rate_of_amplitudes = np.zeros(nn)
rate_of_amplitudes[0] = 0.9 * np.size(index_ampl1) * h
rate_of_amplitudes[8] = 1.4 * np.size(index_ampl9) * h
print('Rate of amplitudes before thinning: ', rate_of_amplitudes)





target_rate = 10      # rate (λ) of the target spike train
cv = 1.8       # coefficiant of variation (cv) of the target spike train 
sigma = np.sqrt(np.log(cv**2 +1))             # analytical expression of σ
mu = -0.5 * np.log(target_rate**2 * (cv**2 +1))      # analytical expression of μ

def hazard(t):
    pdf = lognorm.pdf(t, sigma, np.exp(mu)) # probability density function
    sf = lognorm.sf(t, sigma, np.exp(mu))   # survival function
    return pdf / sf

# define R_m, which is a bit greater than peak value of hazard function.
# With bigger R_m more spikes will be rejected. With smaller R_m, more 
# spikes will be kept, which makes an efficient thinning process.
xs = np.linspace(0, T, 300)
R_m = max(hazard(xs)) * 1.1 # magnification factor 1.1. It can be replaced.
comp_rate = [R_m] * (2 ** (nn) -1)
marked_process3 = pp.generate_markedProcess(comp_rate, T)
correlatedST3 = pp.markedProcess_to_correlatedST(marked_process3) 


new_S =  [np.array([])] * nn
isi_logn = [np.array([])] * nn
for i in range (nn):
    new_S[i] = pp.thinning(correlatedST3[i], target_rate, cv, T)
    isi_logn[i] = np.diff(new_S[i])


counts= [np.array([])]*nn
bin_edges2= [np.array([])]*nn
for i in range(nn):
    count, bin_edge= np.histogram(new_S[i], bins = L, range =(0, T), density= False)
    counts[i]=count
    bin_edges2[i] = bin_edge   
counts = np.sum(counts,axis = 0)

#caculate emprical y values
y_emprical = []
sum1 = np.zeros(sample_size)
for k in range(L):  # L corresponds to n in the formula
    for l in range(L):
        if k < l:
            sum1 = np.cos((counts[k]- counts[l]) *thetas) + sum1
y_emprical =  - np.log(L) + 1/2 * np.log(L + 2 *sum1)

  
# analytical x
x = np.zeros((sample_size, n_v))
for i in range(sample_size):
    for k in range(n_v):
        x[i,k] = np.cos((k+1) * thetas[i]) - 1
        
log_amplitude = np.dot(x, rate_of_amplitudes)
plt.figure(figsize = (20,5) )
plt.plot(thetas, y_emprical, label = 'Emprical Characteristic Func.')  
plt.plot(thetas,log_amplitude, label = 'Analytical Characteristic Func.')   
plt.legend()


from scipy.optimize import lsq_linear
b1 = np.zeros(nn)
#b2 = [np.sum(nr_ones)/L] *nn
b2 = [np.inf]*nn
res= lsq_linear(x, y_emprical, bounds=(b1, b2), lsmr_tol='auto', verbose=1)
res['x']
# pseudo inverse 
lambda_est = np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,y_emprical)) # lambda_est = (X^T X)^(-1) X^T y
print(lambda_est)

plt.figure()
count, bins, ignored = plt.hist(isi_logn, density=True)
xss = np.linspace(min(isi_logn[0]),max(isi_logn[0]), 100)
pdf =lognorm.pdf(xss, sigma, np.exp(mu))
plt.plot(xss,pdf, c = 'k')
