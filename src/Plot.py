# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:34:23 2021

@author: xiuzhi.li@merkur.uni-freiburg.de

This module contians functions to plot spike trains. It has a dependence on
module utils.

* plain_plot
    Plot spike trains in black color. X-axis is observation duration. Y-aixis
    displays number of spike trains.  
* colorPlot_correlatedST
    Plot the correlated spike trains in color. It provides a parameter to set
    same component with same color or same amplitude in the compound process
    with same color. Color map in Python can be prefined. X-axis is  
    observation duration. Y-aixis displays number of spike trains.
"""

import matplotlib.pyplot as plt
import numpy as np
#from src import utils
import utils

def plain_plot(spike_trains):
    """
    Plot the given spike trains in black color.
    :param spike_trains: a list of arrays, each array is a spike train
    :return: 
    """
    n = np.shape(spike_trains)[0]
    plt.figure() 
    for i in range(n):
        x = spike_trains[i]
        y = np.empty(np.size(spike_trains[i]))
        y.fill(i+1) 
        plt.plot(x, y, marker = '|', linestyle = '', markersize=30, c = 'k')
    plt.xlabel('Obervation duration')
    plt.ylabel('Spike trains')
    plt.yticks(np.arange(0, n+2, step=1));
    

def colorPlot_correlatedST(marked_process, color_key = 'components',
                           color_map=plt.cm.hsv):
    """
    Plot correlated spike trains from given marked process in color.
    :param marked_process: a list of two arrays. First array is carrier 
            containing time points of each spike. Second array is integer 
            mark of the component from which the spike come from.
    :param color_key: String type. It can has two values: 
            'components': default. Mark spikes with different color according
             to which component does the spike come from.
            'amplitudes': mark spikes with different color according to 
             amplitudes (number of components), e.g. doubles, triplets.
    :param color_map: a python module, with which spikes are to be displayed  
            in color. Default is matplotlib.cm.hsv.
    :return: 
    """  
    carrier = marked_process[0]         # extract carrier process
    int_mark = marked_process[1]        # extract integer mark of components
    binary_mark = utils.to_binary_matrix(int_mark) # change into binary mark
    number_of_neurons = np.shape(binary_mark)[0]
    # Find indices of the color, same coloums in binary_mark have same color.
    components, indices = np.unique(binary_mark, axis = 1, return_inverse=True)
    # Generate color on the spectrum of input color_map.
    colors = color_map(np.linspace(0, 1, max(indices) + 1, endpoint = False))    
    # Use amplitudes as color indices. Same amplitudes have same color.
    if color_key == 'amplitudes':
        indices = np.sum(binary_mark, axis = 0) - 1
        colors = color_map(np.linspace(0, 1, max(indices)+1, endpoint = False))
    # Plot the spike trains.
    # For each spike in carrier, go through each neuron, if the corresponding 
    # binary mark is 1, plot the spike for this neuron.
    plt.figure() 
    for i in range(len(carrier)):
        for j in range(number_of_neurons):
            x = carrier[i]
            if  binary_mark[j,i]:       # find the binary code
                plt.plot(x, j+1, marker = '|', linestyle = '',
                         markersize = 30, color = colors[indices[i]]) 
    plt.xlabel('Observation duration')
    plt.ylabel('Spike trains');
    plt.ylim([0.5, number_of_neurons + 0.5])
    plt.yticks(np.arange(1, number_of_neurons + 0.5, step=1));
  