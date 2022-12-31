# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:23:57 2021

@author: xiuzhi.li@merkur.uni-freiburg.de

This file is for decompounding the population activity and plot the results using EDP and fitting log amplitude characteristic functions.
"""

import matplotlib.pyplot as plt
import numpy as np 
from src import Correlated_Poisson_Process as pp
from src import utils

def index_of_amplitude(nn, amplitude):
    """
    To find index of specific amlitude in the rate list 
    :param nn: number of neuron
    :param amplitude: a specific order of correlation
    :return: a list of integers.
    """
    def to_binary(number,n):
        """
        Transform given number into binary code of given length. 
        :param number: a decimal number
        :param n: length of the binary code
        :return: binary code.
        """
        return ('{0:0'+str(n)+'b}').format(number)
    
    ampl = []
    
    for i in range(1, 2**nn):
        binary = to_binary(i,nn)
        ampl.append(binary.count("1"))
    ampl = np.array(ampl)
    return np.where(ampl==amplitude)

# 
# Input parameter is a list of integers.
def count_ones(n):
    '''
    Count how many ones in the binary code for an integer.
    :param n: an array of integer numbers
    :return: an array of integers, concieving how many ones.
    '''
    ones = np.zeros(len(n), dtype = int)
    for i in range(len(n)):
        ones[i] = bin(n[i]).count("1")  
    return ones

#set parameters
T = 100       # observation duration
sample_size =100     # smaple size of theta
thetas = np.arange(- np.pi, np.pi, step = 2 * np.pi / sample_size)
nn = 10       # analytical number of neurons
L = 1000       # nubmer of bin
h = T/L      # bin size
n_v = 10


# define true amplitude distribution
index_ampl1 = index_of_amplitude(nn, 1)
index_ampl2 = index_of_amplitude(nn, 2)
index_ampl3 = index_of_amplitude(nn, 3)
index_ampl5 = index_of_amplitude(nn, 5)
index_ampl9 = index_of_amplitude(nn, 9)
rate = np.zeros(2 ** nn -1)
rate[index_ampl1]= 0.4 
rate[index_ampl2]= 0.5
rate[index_ampl3]= 0.6 
rate[index_ampl5]= 0.7 
rate[index_ampl9]= 1

rate_of_amplitudes = np.zeros(nn)
rate_of_amplitudes[0] = 0.4 * np.size(index_ampl1) * h
rate_of_amplitudes[1] = 0.5 * np.size(index_ampl2) * h
rate_of_amplitudes[2] = 0.6 * np.size(index_ampl3) * h
rate_of_amplitudes[4] = 0.7 * np.size(index_ampl5) * h
rate_of_amplitudes[8] = 1 * np.size(index_ampl9) * h
print('Expected rate of amplitudes: ', rate_of_amplitudes/h)

# Stimulation empirical data
marked_process = pp.generate_markedProcess(rate, T)
correlatedST2 = pp.markedProcess_to_correlatedST(marked_process)  

# caculate counts via marked prococess
nr_ones = count_ones(marked_process[1])
counts, bin_edges= np.histogram(marked_process[0], bins = L, range =(0, T), weights = nr_ones, density= False)

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

# linear fitting with constraints, scipy.optimize.lsq_linear
from scipy.optimize import lsq_linear
b1 = np.zeros(n_v)
b2 = [np.sum(nr_ones)/L] *n_v
#b2 = [np.inf]*nn
res= lsq_linear(x, y_emprical, bounds=(b1, b2), lsmr_tol='auto', verbose=1)
#res['x']/h

# pseudo inverse 
lambda_est = np.dot(np.linalg.pinv(np.dot(x.T,x)),np.dot(x.T,y_emprical)) # lambda_est = (X^T X)^(-1) X^T y
#print(lambda_est/h)

# EDP - analytical estimated rates, using the foluma in the paper, same as obove section.
import scipy.integrate as integrate
gamma_est = []
v = np.zeros(n_v)
z_l =[0] * L + 1j * counts
for tt in thetas:
    gamma_est.append(1/L * np.sum(np.e ** (z_l * tt)))

n_l =[0] *n_v - 1j * (np.arange(n_v)+1)    

def func(t, k): # t is theta
    return np.log( 1/ L * np.sum (np.e ** (z_l * t)) ) * np.e**(n_l[k] * t)

for k in range(n_v):
    integral = integrate.quad(func, -np.pi, np.pi, args = (k))
    v[k] = 1 / (2 * np.pi) * (1 / h) * integral[0]
#print(v)

log_amplitude = np.dot(x, res['x'])
# Plot both functions
plt.figure(figsize = (20,5) )
plt.plot(thetas, y_emprical, label = 'Emprical Characteristic Func.')  
plt.plot(thetas,log_amplitude, label = 'Analytical Characteristic Func.')   
plt.legend()

# plot amplitude distribution of emprical data, analytical and final result
int_mark = marked_process[1]
binary_mark = utils.to_binary_matrix(int_mark)
amplitudes = np.sum(binary_mark, axis = 0)
ampl, occurence = np.unique(amplitudes, return_counts=True)
empirical_rates = occurence/T

index = np.arange(1, len(rate_of_amplitudes)+1)
index_cal = np.arange(1, len(lambda_est)+1)
w = 0.2

plt.figure()
plt.bar(index-w, rate_of_amplitudes/h, width=0.2, align='center', alpha = 0.8, label= 'Desired distr.')
plt.bar(ampl, empirical_rates, width=0.2, align='center', alpha = 0.8, label = 'Emprical distr.')
plt.bar(index_cal+w, lambda_est/h, width=0.2, align='center', alpha = 0.8, label= 'Log Ampl.C.F.')
plt.bar(index_cal+ 2*w, v, width=0.2, align='center', alpha = 0.8, label= 'EDP')
plt.xticks(index_cal)
plt.ylim([min(empirical_rates)-0.25, max(empirical_rates)+10])
plt.xlabel('Amplitudes')
plt.ylabel('Rate [Hz]')
plt.title('Using Moore Penrose inverse')
plt.legend()

plt.figure()
plt.bar(index-w, rate_of_amplitudes/h, width=0.2, align='center', alpha = 0.8, label= 'Desired distr.')
plt.bar(ampl, empirical_rates, width=0.2, align='center', alpha = 0.8, label = 'Emprical distr.')
plt.bar(index_cal+w, res['x']/h, width=0.2, align='center', alpha = 0.8, label= 'Log Ampl.C.F.')
plt.bar(index_cal+ 2*w, v, width=0.2, align='center', alpha = 0.8, label= 'EDP')
plt.xticks(index_cal)
plt.ylim([min(empirical_rates)-0.25, max(empirical_rates)+10])
plt.xlabel('Amplitudes')
plt.ylabel('Rate [Hz]')
plt.title('Using linear fitting with constraints')
plt.legend()
