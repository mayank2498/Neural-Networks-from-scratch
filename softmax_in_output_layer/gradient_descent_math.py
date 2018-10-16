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
        self.alpha = 3
        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward_prop(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        return sigmoid(self.z3)
    
    def getCost(self,y):
        print("Cost : ", 0.5*sum((y-self.yhat)**2) )
    
    def derivatives_J_wrt_W(self,X,y):
        
        # delta3 is of size n x 1 , where n is number of training points
        delta3 = -1*np.multiply(y-self.yhat, sigmoid_derivative(self.z3))
        
        dJdW2 = np.dot(self.a2.T,delta3)
        
        delta2 = np.dot(delta3,self.W2.T) * sigmoid_derivative(self.z2)
        dJdW1 = np.dot( X.T , delta2 )  
        
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
X = np.array([[0,1],[1,0],[1,1],[0,0]])
y = np.array([[1],[1],[0],[1]])

             
nn.train(X,y)
nn.train(X,y)


epochs = 5000
for i in range(0,epochs):
    nn.train(X,y)