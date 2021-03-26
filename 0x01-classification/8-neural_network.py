#!/usr/bin/env python3
import numpy as np


class NeuralNetwork():
    

    def __init__(self, nx, nodes):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.W1 = np.random.randn(nodes,nx)*0.01)
            self.b1 = np.array([[np.array(0.)]] * nodes)
            self.A1 = 0
            self.W2 = np.random.randn(1,nodes)*0.01)
            self.b2 = 0
            self.A2 = 0
