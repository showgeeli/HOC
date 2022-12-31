# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:10:14 2021

@author: xiuzhi.li@merkur.uni-freiburg.de

This module is to generate correlated spike trains via Poisson processes. It
enables to create parallel spike trains as components, and using the
methode proposed in Staude et al., 2010 (see reference [2]) to generate spike 
trains for neurons with higher-order correlations. Besides, it also enables to 
create correlated spike trains via pre-defined rate distribution and amplitude 
distribution. Further, a thinning process - a renewal process with harzard 
function (reference [3]) is provided. Via thinning process, some spikes will 
be rejected, so that the inter-spike-interval (ISI) distribution follows a 
lognormal distribution.

* generate_parallelST
    Generate spike train(s) via stationary Poisson process. 
* parallelST_to_markedProcess
    Use parallel spike trains as components to create marked process.
* markedProcess_to_components
    Transform marked processes into components (parallel spike trains).
* generate_markedProcess
    Generate marked process from a list of rates and obervation duration.
* direct_markedProcess
    Produce markedProcess directly via rate distribution of components.
* markedPrcocess_to_correlatedST
    Transform markred process into spike trains with higher-order correlations.
* generate_popuActivity
    Generate correlated spike train via pre-defined amplitude distribution.
* thinning
    Via a renewal process with harzard function, some spikes will be rejected. 
    After thinning, ISI distribution follows lognormal distribution.

References
----------
.. [1] Kuhn A, Aertsen A, Rotter S. Higher-order statistics of input ensembles 
   and the response of simple model neurons. Neural Comput. 2003 Jan;
   15(1):67-101. doi: 10.1162/089976603321043702.
.. [2] Benjamin Staude, Sonja Grün, and Stefan Rotter. Higher-Order 
   Correlations and Cumulants. In Sonja Grün and Stefan Rotter, editors, 
   Analysis of Parallel Spike Trains, pages 253–280. Springer US, 2010.
.. [3] Reimer IC, Staude B, Ehm W, Rotter S. Modeling and analyzing 
   higher-order correlations in non-Poissonian spike trains. J Neurosci 
   Methods. 2012 Jun 30;208(1):18-33. doi: 10.1016/j.jneumeth.2012.04.015.
"""

import numpy as np
from itertools import chain
#from src import utils
import utils as utils
from scipy.stats import lognorm

def generate_parallelST(lambdas, T):
    """
    Generate single or multiple spike trains via stationary Poisson process 
    with specific rates and observation duration. 
    :param lambdas: a list of rates
    :param T: observation duration
    :return: a list of numpy arrays. Each array is one spike train.
    """
    def generate_one(rate):
        """
        Generate one spike train via stationary Poisson process. Steps are:
            1. sample a random number k from P(λT)
            2. distribute k spikes uniformly and independently on [0,T)
            3. spike train is collection of all points.       
        :param rate: rate of the spike train to be generated   
        :return: a numpy array of spikes
        """
        k = np.random.poisson(rate * T) #draw a random number from P(lambda0*T)
        spike_train = np.random.uniform(0, T, k) #distribute k spikes on [0,T)
        spike_train.sort()  # sort the spikes 
        return spike_train   
    
    spike_trains = []
    # generate spike trains for each rate
    for rate in lambdas:
        spike_trains.append(generate_one(rate))    
    return spike_trains

def parallelST_to_markedProcess(spike_trains):
    """
    Use parallel spike trains as components to create marked process.
    :param spike_trains: list of numpy arrays, containing parallel spike trains 
    :return: list of two numpy arrays. First array is carrier process 
            containing spikes of all neurons. Second array is integer mark of 
            component from which the spike comes from.
    """
    if np.log2(len(spike_trains) + 1) % 1 != 0:
       raise ValueError(" The number of spike trains should be 2ᴺ-1. "
                        "N is the number of neurons.")
    else:  
        # Merge all spike trains into one carrier process.
        carrier = np.fromiter(chain.from_iterable(spike_trains), float) 
        carrier.sort()  # sort time points in carrier
        int_mark = np.zeros(len(carrier), dtype = np.int64)
        # Use index of the component + 1 as an integer mark for each spike 
        # in carrier process.
        for i, spike_train in enumerate(spike_trains): 
            idx = np.searchsorted(carrier, spike_train) 
            int_mark[idx] = i + 1            
        return [carrier, int_mark] 
    

def  markedProcess_to_components(marked_process):  
    """
    Transform marked processes into components (parallel spike trains).
    :param marked_process: a list of two arrays. First array is a carrier
            process. Second array is integer mark.
    :return: list of numpy arrays. Each array is a component.
    """  
    carrier = marked_process[0]             # extract carrier
    int_mark = marked_process[1]            # extract integer mark
    nn = int(max(int_mark)).bit_length()    # nn is number of neurons
    # There should be 2ᴺ-1 components.
    components = [np.array([])] * (np.power(2,nn) - 1) 
    # Put each spike in carrier into the component it belongs to. The integer 
    # number - 1 is index of the corresponding component. 
    for n in range(len(carrier)):
        idx = int(int_mark[n]-1) 
        components[idx] = np.hstack((components[idx], carrier[n]))
    return components  

    
def generate_markedProcess(lambdas, T):
    """
    Generate marked process from a list of rates and obervation duration. The 
    number of rates should be 2ᴺ-1. N is the number of neurons. 
    :param lambdas: a list of rates. If the number of rates does not equal to 
            2ᴺ-1 (N is the number of neurons), an error message is thrown.
    :return: list of two numpy arrays. First array is carrier process 
            containing spikes of all neurons. Second array is integer mark of 
            component from which the spike comes from.
    """  
    
    if np.log2(len(lambdas) + 1) % 1 != 0:
        raise ValueError("The number of rates should be 2ᴺ-1. "
                         "N is the number of neurons.")
    else:  
        components = generate_parallelST(lambdas, T)        
        return parallelST_to_markedProcess(components) 
    
    
def direct_markedProcess(lambdas, T):
    """
    Produce markedProcess directly via rate distribution of components.
    :param lambdas: a list of rates. If the number of rates does not equal to 
            2ᴺ-1 (N is the number of neurons), an error message is thrown.
    : param T: observation duration
    :return: list of two numpy arrays. First array is carrier process 
            containing spikes of all neurons. Second array is integer mark of 
            component from which the spike comes from.
    """
    if np.log2(len(lambdas) + 1) % 1 != 0:
        raise ValueError("The number of rates should be 2ᴺ-1. "
                         " N is the number of neurons.")
    else:
        #simulate carrier process with the sum of all component rates
        carrier = generate_parallelST([np.sum(lambdas)], T)[0] 
        pattern_possib = lambdas/np.sum(lambdas) # create rate distribution
        # np.arange(0, len(lambdas)) as the index of rates (lambdas).
        # For each spike in carrier, draw an index of rates randomly based
        # on rate probability distribution, plus one. Take it as integer mark.
        int_mark = np.random.choice(np.arange(0, len(lambdas)), len(carrier),
                                    p = pattern_possib) + 1
        return [carrier, int_mark]


def markedProcess_to_correlatedST(marked_process):
    """
    Transform markred process into spike trains with higher-order correlations.
    :param marked_process: list of two numpy arrays. First array is carrier 
            process containing spikes of all neurons. Second array is integer 
            mark of component from which the spike comes from.
    :return: list of numpy arrays, each array is a spike train of one neuron.
            They are correlated in higher orders.
    """
    carrier = marked_process[0]       # extract carrier
    int_mark= marked_process[1]       # extract integer mark
    binary_mark = utils.to_binary_matrix(int_mark) # change to binary marks
    nn = np.shape(binary_mark)[0]   # nn is the number of neurons

    correlatedST = [np.array([])] * nn  # initialize correlated spike trains
    # For each neuron, find where binary mark is 1. The corresponding time in
    # carrier is one spike of that neuron.  
    for i in range(nn):
        correlatedST[i] = np.array(carrier[np.where(binary_mark[i,:]==1)])  
        
    return correlatedST   

def generate_popuActivity(amplitudes_dist, carrier_rate, T):
    '''
    
    A new function to generate population spike trains via pre-defined 
    amplitude, distribution rate of carrier and observation period.
    :param amplitudes_dist: a list of amplitude distubition, normalized rates.
                            Sum of all values equals 1.
    :param carrier_rate: float, a rate used to stimulate carrier process.
    :param T: observation duration
    :return: list of numpy arrays, each array is a spike train of one neuron.
            They are correlated with higher orders.
    '''
    nn = len(amplitudes_dist) # number of neurons
    correlatedST = [np.array([])] * nn  # initialze spike trains
    carrier = generate_parallelST([carrier_rate], T)[0]#realize carrier process
    
    # For each spike in carrier, draw a number of amplitudes according to 
    # probability distribution. Assign the spike uniformly to that number 
    # (amplitude_drawn) of neuron.
    for idx, spike in enumerate(carrier):
        amplitude_drawn= np.random.choice(np.arange(nn), 1,   
                            p = amplitudes_dist) +1           
        neuron_ids = np.random.choice(np.arange(nn), 
                                      amplitude_drawn, replace=False) 
        for nid in neuron_ids:                                                     
            correlatedST[nid]= np.append(correlatedST[nid],spike)                      
    return correlatedST

def thinning(S, rate, cv, T):
    """
    An input spike train will be thinned. Some spikes are deleted so that the
    ISI distribution follows a lognormal distribution instead of exponential
    distribution.
    :param S: the spike train to be thinned
    :param rate: rate of target spike train
    :param cv: coefficient of variation of target spike train
    :param T: observation period
    :return: an array of spikes
    """
    # Target spike train is a lognormal process. 
    # Caculate σ and μ according to rate and cv.
    sigma = np.sqrt(np.log(cv**2 +1))        # analytical expression of σ
    mu = -0.5 * np.log(rate**2 * (cv**2 +1)) # analytical expression of μ
  
    def hazard(t):
        pdf = lognorm.pdf(t, sigma, np.exp(mu)) # probability density function
        sf = lognorm.sf(t, sigma, np.exp(mu))   # survival function
        return pdf / sf
    
    # define R_m, which is a bit greater than peak value of hazard function.
    # With bigger R_m more spikes will be rejected. With smaller R_m, more 
    # spikes will be kept, which makes an efficient thinning process.
    x = np.linspace(0, T, 300)
    R_m = max(hazard(x)) * 1.1 # magnification factor 1.1. It can be replaced.

    #def renewal(S, hazard, R_m):
    n = len(S)
    U = np.random.uniform(0,R_m, n) # draw n random values between [0, R_m)
    t_latest = S[0] 
    new_S = [t_latest]
    # For each spike draw a random value between [0, R_m). If the random value
    # greater than the value of hazard function, reject the spike and continue
    # the hazard function until the random value is no greater than the hazard 
    # function. Then save the spike and reset the hazard function.
    for i in range(1,n):
        if U[i] <= hazard([S[i]-t_latest]):
            t_latest = S[i]
            new_S.append(t_latest)
    return new_S  