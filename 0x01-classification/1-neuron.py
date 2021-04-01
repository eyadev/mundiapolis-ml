#!/usr/bin/env python3
'''Neuron class'''

import numpy as np


class Neuron:

    def __init__(self, nx):
        '''Initialization function for the Neuron class'''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''return W value'''
        
        return self.__W

    @property
    def b(self):
        '''Return b value'''

        return self.__b

    @property
    def A(self):
        '''return A value'''

        return self.__A