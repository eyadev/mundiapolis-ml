#!/usr/bin/env python3
'''  Neuron class  '''

import numpy as np


class Neuron:
    ''' class for initialising the neuron '''

    def __init__(self, nx):
        ''' Neuron function '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0