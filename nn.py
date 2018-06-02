import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
    
def sigmoid_derivative(x): 
    return x * (1 - x)
    
    
    
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)*0.1
        print("Weights 1 : ",self.weights1)
        self.weights2   = np.random.rand(4,1)*.01
        print("Weights 2 : ",self.weights2)               
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def forwardprop(self):
        # create a hidden layer named layer 1
        
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        print("Layer 1 ",self.layer1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        print("Output layer ",self.output)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
Input = np.array([[0.0023,0.1231,0.0004]])
output = np.array([[1]])

network = NeuralNetwork(Input,output)


for i in range(1000):
    network.forwardprop()
    network.backprop()
    print("Epoch ",i," is done \n\n")