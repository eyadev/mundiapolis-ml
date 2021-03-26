#!/usr/bin/env python3
import numpy as np


class Neuron:


    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.randn(1,2)*0.01 
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        self.__A = self.sigmoid(
            np.matmul(self.__W, X) + self.__b
        )
        return self.__A

    def sigmoid(self, X):
        return 1.0/(1.0 + np.exp(-X))

    def cost(self, Y, A):
        m = A.shape[1]
        cost = (
                -(1 / m)
            ) * (np.sum(
                    (
                        Y * np.log(A)
                    ) + ((
                            1 - Y
                        ) * np.log(
                            1.0000001 - A
                        )
                    )
                )
            )
        return cost

    def evaluate(self, X, Y):
        propagation = self.forward_prop(X)
        self.__A = np.where(propagation >= 0.5, 1, 0)
        cost = self.cost(Y, propagation)
        return (self.__A, cost)
