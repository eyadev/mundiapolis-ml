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
        return 1 / (1 + np.exp(-X))

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
        self.__A = self.forward_prop(X)
        return (
            np.where(self.__A >= 0.5, 1, 0),
            self.cost(Y, self.__A)
        )

    def gradient_descent(self, X, Y, A, alpha=0.05):
        self.__W = np.add(
            self.W,
            -alpha * np.matmul(
                A - Y,
                X.T
            ) / X.shape[1]
        )
        self.__b += np.mean(
                (A - Y)
            ) * -alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        [
            self.gradient_descent(
                X, Y, self.forward_prop(X), alpha
            ) for _ in range(iterations)
        ]
        return self.evaluate(X, Y)
