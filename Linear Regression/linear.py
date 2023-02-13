# import the libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load the dataset
dataset = pd.read_csv("dataset/Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/6, random_state=0)

# Fit the simple linear regression model to the training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction of test and training set result
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

# visualize the train set result
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary vs Experience (Training Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(INR)")
plt.show()

# visualizing the test set results
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, y_pred, color="red")
plt.title("Salary vs Experience (Test Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(INR)")
plt.show()
