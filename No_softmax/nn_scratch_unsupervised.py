##  Buggy code 

##  Value of Cost increases on every iteration
##  Equations have to be checked again


import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-1*z))
    
def sigmoid_derivative(z):
    return np.exp(-1*z)/ (1+np.exp(-1*z)**2 )
    
class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.alpha = 0.8
        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward_prop(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        # NOTE: No sigmoid here !
        return self.z3
    
    def getCost(self,y):
        print("Cost : ", 0.5*sum((y-self.yhat)**2) )
    
    def derivatives_J_wrt_W(self,X,y):
        
        dJdW2 = -1*np.dot(self.a2.T,y-self.yhat)
        x = -1*np.dot(y-self.yhat,self.W2.T)
        y = np.multiply(sigmoid_derivative(self.z2),x)
        dJdW1 = np.dot(X.T,y)
        
        dJdW1 = np.random.randn(2,3)
        return dJdW1 , dJdW2
    
    def backpropagate(self,X,y):
        dJdW1 , dJdW2 = self.derivatives_J_wrt_W(X,y)
        self.W1 = self.W1 - self.alpha * dJdW1
        self.W2 = self.W2 - self.alpha * dJdW2
        
    
    def train(self,X,y):
        self.yhat = self.forward_prop(X)
        self.backpropagate(X,y)
        self.getCost(y)
        
nn = Neural_Network()
X = np.array([[1,2],[2,3],[4,5]])
y = np.array([[3],[5],[9]])

             
nn.train(X,y)
nn.train(X,y)


epochs = 10
for i in range(0,epochs):
    nn.train(X,y)
