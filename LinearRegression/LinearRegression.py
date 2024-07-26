import numpy as np

class LinearRegreesion:
    def __init__(self,iter=100,alpha=0.001):
        self.iter=iter
        self.alpha=alpha
        self.weight=None
        self.bias=None
    
    def fit(self,X,y):
        m,features=X.shape
        self.weight=np.zeros(features)
        self.bias=0

        for _ in range(self.iter):
            y_pred=np.dot(self.weight,features)+self.bias

            change_in_weight=(1/m)*np.dot(y_pred-y,X.T)
            change_in_bias=np.sum(y_pred-y)

            self.weight=self.weight-self.alpha*(change_in_weight)
            self.bias=self.bias-self.alpha*(change_in_bias)
    
    def predict(self,X):
        y_pred=np.dot(X,self.weight)+self.bias
        return y_pred