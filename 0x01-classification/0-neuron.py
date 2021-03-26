#!/usr/bin/env python3
import numpy as np


class Neuron:

    
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:    
        self.w=np.random.randn(1,2)*0.01 
        self.b=0
        self.A=0   
