import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self,epochs=1000,alpha=0.001):
        self.epochs=epochs
        self.alpha=alpha
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature)
        self.bias=0

        for _ in range(self.epochs):
            linear=np.dot(self.weights,X)+self.bias
            y_pred=sigmoid(linear)

            dw=(1/n_sample)*np.dot(X.T,(y_pred-y))
            db=np.sum((y_pred-y))
    
    def predict(self,X):
        return sigmoid(np.dot(self.weights,X)+self.bias)

