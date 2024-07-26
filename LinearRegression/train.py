import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn import datasets

X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=20,random_state=42)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

def error(y_pred,y_actual):
    return np.mean((y_actual-y_pred)**2)

mse=error(y_pred,y_test)
print(mse)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()