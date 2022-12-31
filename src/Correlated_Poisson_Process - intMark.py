# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:10:14 2021

@author: xiuzhi.li@merkur.uni-freiburg.de
"""


import numpy as np
from itertools import chain
from scipy.stats import lognorm

def generate_parallelST(lambda0, T, n=1):
    """
    generate n spike trains via stationary poisson process with (lambda0 * T)
    :param lambda0: rate
    :param T: observation duration
    :param n: the number of spkie trains to be generated, default is 1.
    :return: spike trains, a list of arrays
    """
    def generate_one():
        """
        generate single spike train via stationary poisson process with (lambda0 * T)
        :return: one spike train a numpy array,
        """
        k = np.random.poisson(lambda0 * T)  # sample random number k from P(lambda0*T); lambda0*T is the expected number of spikes
        spike_train = np.random.uniform(0,T,k) # distribute k spikes uniformly on [0,T)
        spike_train.sort()  # sort the spikes
        return spike_train
    
    return [generate_one() for i in range(n)]

def to_binary(number,n):
    return ('{0:0'+str(n)+'b}').format(number)

def to_binary_matrix(int_array):
    n_row = max(int_array).bit_length()
    n_col = len(int_array) 
    binary_mark = np.zeros(n_row,n_col)
    for i in range(n_col):
        binary_mark[i] = to_binary(int_array[i],n_row)
    return binary_mark


def parallelST_to_markedProcess(spike_trains):
    """
    Transform parallel spike trains into marked processes.
    :param spike_trains: list of arrays, each array is a parallel poissonial spike train. 
                         These spike trains are taken as components of 1st, 2nd, 3rd.... order in sequence.
    return: list of two arrays. One array is carrier containing time points of pikes. 
            Another array is binary matrix of neuron spikes.
    """

    if np.log2(len(spike_trains) + 1) % 1 != 0:
       print ('The number of spike trains should be 2ᴺ-1. N is the number of neurons.')
    else:  
        carrier = np.fromiter(chain.from_iterable(spike_trains), float) # merge all spike trains into one carrier process
        carrier.sort() # sort time points in carrier
        component_int_mark = np.zeros(len(carrier))
        for i, spike_train in enumerate(spike_trains): 
            idx = np.searchsorted(carrier,spike_train) 
            component_int_mark[idx] = i+1
            
        return [carrier, component_int_mark] 
    

def  markedProcess_to_components(marked_processes):  
    """
    Transform the input marked processes into components, which are parallel spike trains.
    :param marked_process: a list of two arrays. The first array is a carrier spike train.
                            Second array is a binary matrix.
    return: list of arrays. Each array is a component of the marked processes. 
   
    """  
    carrier = marked_processes[0]   
    component_int_mark = marked_processes[1]
    nn = int(max(component_int_mark)).bit_length()          # n is number of neurons
    components = [np.array([])] * (np.power(2,nn) - 1)      # initialize the shape of the array of spike trains

    for n in range(len(carrier)):
        idx = int(component_int_mark[n]-1) # The integer number in component_int_mark minus 2 is the index of corresponding component
        components[idx] = np.hstack((components[idx], carrier[n]))
    return components  
    
def generate_markedProcess(lambdas, T):
    """
    Generate marked process of correlated poisson process from a list or an array of rates and obervation time.
    :param lambdas: a list or an array of rates. If the number of rates does not equal to 2ᴺ-1 (N is the number of neurons), 
                    an error message will be printed. The input rates will be acknowledged as rates 
                    of 1st, 2nd, 3rd.... order of components in sequence.
    return: list of two arrays. One array is carrier containing time points of pikes. 
            Another array is binary matrix of neuron spikes.
    """  
    
    if np.log2(len(lambdas) + 1) % 1 != 0:
        print ('The number of rates should be 2ᴺ-1. N is the number of neurons.')
    else:  
        components = []        
        for rate in lambdas:
            components.extend(generate_parallelST(rate, T))          
        return parallelST_to_markedProcess(components) 


def markedProcess_to_correlatedST(marked_process):
    """
    Transform makred processes into correlated spike trains.
    :param marked_process: list of two arrays. First array is carrier containing time points of pikes. 
                             Second array is binary matrix of neuron spikes.
    return: list of arrays, each array is a spike train of a neuron.
    """
    carrier = marked_process[0]                 # get carrier
    component_int_mark= marked_process[1]       # get integer mark of components id
    nn = max(component_int_mark).bit_length()   # nn is the number of neurons
    correlatedST = [np.array([])] * nn          # initialize correlated spike trains
    binary_mark = to_binary_matrix(component_int_mark)        # change integer mark into binary marks
    
    # Find index of each row where binary mark is 1. The corresponding time in carrier is one spike of that neuron.  
    for i in range(nn):
        correlatedST[i] = np.array(carrier[np.where(binary_mark[i,:]==1)])  
        
    return correlatedST   



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
    return new_S  


