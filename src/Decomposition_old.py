# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:47:18 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""
import matplotlib.pyplot as plt
import numpy as np 
from src import Correlated_Poisson_Process as pp
from src import utils
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


T = 10       # observation duration
sample_size =100     # smaple size of theta
thetas = np.arange(-np.pi, np.pi, step = 2 * np.pi / sample_size)

# generate poplulation spike trains methode 2: via component rates
L = 50       # nubmer of bin
h = T/L      # bin size
n_v = 10
nn = 10       # analytical number of neurons
index_ampl1 = index_of_amplitude(nn, 1)
index_ampl9 = index_of_amplitude(nn, 9)
rate = np.zeros(2 ** nn -1)
rate[index_ampl1]= 0.9 
rate[index_ampl9]= 0.1
print('Expected rate of amplitude 1: ', 0.9 * h * np.size(index_ampl1))
print('Expected rate of amplitude 9: ', 0.1 * h * np.size(index_ampl9))

marked_process2 = pp.generate_markedProcess(rate, T)
correlatedST2 = pp.markedProcess_to_correlatedST(marked_process2)    

count1, bins1, patches1 = plt.hist(correlatedST2,L, density=False, stacked=True, align='mid', alpha = 0.8)
counts = np.sum(count1, axis =0)   # bin counts
plt.clf()
#print(counts)
counts2, bins2, patches2 = plt.hist(counts,L, density= False, stacked=True, align='mid', alpha = 0.8)


# plot amplitude distribution of emprical data
int_mark = marked_process2[1]
binary_mark = utils.to_binary_matrix(int_mark)
amplitudes = np.sum(binary_mark, axis = 0)
plt.figure()
count_ampl, bins_ampl, patches_ampl = plt.hist(amplitudes,L, density=False, stacked=True, align='mid', alpha = 0.8)

        

# EDP - analytical estimated rates, using the foluma in the paper
# Integrate real and imag parts seperately.
import scipy
from scipy.integrate import quad
gamma_est = []
v = np.zeros(n_v)
z_l =[0] * L + 1j * counts2
    
n_l =[0] *n_v - 1j * (np.arange(n_v)+1)    

def func(t, k): # t is theta
    return np.log( (1/ L) * np.sum (np.e ** (z_l * t)) ) * (np.e**(n_l[k] * t) )

def real_func(t, k):
    return scipy.real(func(t, k))
def imag_func(t, k):
    return scipy.imag(func(t, k))

for k in range(n_v):

    real_integral = quad(real_func, -np.pi, np.pi, args = (k))  
    imag_integral = quad(imag_func, -np.pi, np.pi, args = (k))
    #print(real_integral, "real")
    #print(imag_integral, "imag")
    integral = real_integral[0] + 1j*imag_integral[0]
    integral = scipy.real(integral)
    v[k] = 1 / (2 * np.pi) * (1 / h) * integral

print(v)




# a script, to check the number of amplitudes for given number of neurons, if the population 
# activity is genereated via component rates. 
from itertools import combinations
number_of_neurons = 10
a = np.arange(number_of_neurons)
for i in range(1,number_of_neurons + 1):
    list1 = list(combinations(a,i))
    print('Amplitude of ',np.shape(list1)[1],  ', Number:', len(list1), '\n')