import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = range(0,9)
y = [3, 7, 10, 24, 50, 95, 50, 24, 10]
plt.scatter(x, y)
plt.show()
x_train = np.linspace(0, 5, 100)
x_test = np.linspace(0, 5, 100)
y_train = x_train * x_train
y_test = x_test * x_train
y_test.reshape(-1, 1)
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.show()
model = LinearRegression()
model.fit(x_train, y_train)
print(model.predict(x_train))
print(model.predict(x_test))
