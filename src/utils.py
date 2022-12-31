# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:27:14 2021

@author: xiuzhi.li@merkur.uni-freiburg.de

Common functions that are used in other modules.

* to_binary
    Change an integer into binary code with specific length.
* to_binary_matrix
    Transform a 1D array of integers into a 2D array of binary code. 
    
"""
import numpy as np

def to_binary_matrix(int_array):
    """
    Transform a 1D array of integers into a 2D array of binary code. 
    The binary code of maximum integer is supposed to contain only 1s (a common 
    component for all neurons). That's why the bit length the maxium integer 
    number can represent number of rows in the 2D array.
    :param int_array: an array of integers
    :return: a 2D array with binary codes. Number of rows corresponds to number
             of neurons. Number of columns corresponds to length of the carrier.
    """
    
    def to_binary(number,n):
        """
        Transform a deimal number into binary code with specific length.
        :param number: an integer number
        :param n: length of binary code. 0 is added on the left side if needed.
        :return: a string of binary code with length n.
        """
        return ('{0:0'+str(n)+'b}').format(number)
    
    n_row = int(max(int_array)).bit_length()
    n_col = len(int_array) 
    binary_mark = np.zeros((n_row,n_col), dtype = np.int8)
    for i in range(n_col):
        binary_mark[:,i] = list(to_binary(int_array[i],n_row))
    return binary_mark

