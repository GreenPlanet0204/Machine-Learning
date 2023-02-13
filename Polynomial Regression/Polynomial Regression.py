# import the necessary package
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
dataset = pd.read_csv('dataset/Position_Salaries.csv')

# extract independent and dependent variable
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fit the linear regression to the dataset
lin_regressor = LinearRegression()
lin_regressor.fit(x, y)

# fit the polynomial regression to the dataset
poly_regressor = PolynomialFeatures(degree=3)
x_poly = poly_regressor.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# visualize the result for Linear Regression model
plt.scatter(x, y, color="blue")
plt.plot(x, lin_regressor.predict(x), color="red")
plt.title("Bluff detection model(Linear Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

# visualize the result for Polynomial Regression
plt.scatter(x, y, color="blue")
plt.plot(x, lin_reg_2.predict(poly_regressor.fit_transform(x)), color="red")
plt.title("Bluff detection model(Polynomial Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

lin_pred = lin_regressor.predict([[6.5]])
print(lin_pred)
poly_pred = lin_reg_2.predict(poly_regressor.fit_transform([[6.5]]))
print(poly_pred)
