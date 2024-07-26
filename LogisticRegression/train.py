import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import seaborn as sb
from sklearn.metrics import confusion_matrix as cm

data = datasets.load_breast_cancer()
X, y = data.data, data.target


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


lr = LogisticRegression()
lr.fit(x_train, y_train)


y_pred = lr.predict(x_test)

plt.figure(figsize=(12, 6))


feature_index = 0
plt.scatter(x_test[:, feature_index], y_test, color='blue', label='Actual')
plt.scatter(x_test[:, feature_index], y_pred, color='red', label='Predicted', marker='x')


plt.xlabel(data.feature_names[feature_index])
plt.ylabel('Target')
plt.legend()
plt.show()

cof_mat=cm(y_test,y_pred)
sb.heatmap(cof_mat,annot=True)
plt.show()