#!/usr/bin/env python3
'''  Neuron class  '''

import numpy as np


class Neuron:
    ''' class for initialising the neuron '''

    def __init__(self, nx):
        ''' Initialization function for the Neuron class '''

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        ''' Function that calculates the forward propagation of the neuron '''

        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

    def cost(self, Y, A):
        ''' Calculates the cost of the model using logistic regression '''
        
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

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
