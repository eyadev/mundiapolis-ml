import numpy as np

Neuron = __import__('0-neuron').Neuron

lib_train = np.load('Bureau/ML/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

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

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)