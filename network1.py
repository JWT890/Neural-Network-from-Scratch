import numpy as np
from numpy import random

# This one made with assistance from Bard

np.random.seed(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, x):
        # Hidden layer
        z1 = np.dot(self.W1, x) + self.b1
        a1 = sigmoid(z1)

        # Output layer
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = sigmoid(z2)

        return a2
    
nn = NeuralNetwork(2, 3, 1)

X = np.array([[0,0], [0,1], [1,0], [1,1]])

for x in X:
    output = nn.forward(x)
    print(output)
