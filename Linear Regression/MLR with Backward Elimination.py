# import the necessary packages
import statsmodels.api as sm  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
dataset = pd.read_csv("../dataset/50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Categorical data
label_encoder_x = LabelEncoder()
x[:, 3] = label_encoder_x.fit_transform(x[:, 3])
onehot_encoder = ColumnTransformer(
    [("State", OneHotEncoder(), [3])], remainder="passthrough")
x = onehot_encoder.fit_transform(x)

# avoiding the dummy variable trap
x = x[:, 1:]

# Splitting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# fit our MLR model to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Train Score:", regressor.score(x_train, y_train))
print("Test Score: ", regressor.score(x_test, y_test))

# Preparation of Backward Elimation
x = np.append(arr=np.ones((50, 1)).astype(int), values=x,
              axis=1)  # add a column in a matrix of features
x = np.array(x, dtype=float)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_BE = x_opt
y_BE = y

x_BE_train, x_BE_test, y_BE_train, y_BE_test = train_test_split(
    x_BE, y_BE, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(x_BE_train, y_BE_train)
y_pred = regressor.predict(x_BE_test)
print("Train Score:", regressor.score(x_BE_train, y_BE_train))
print("Test Score: ", regressor.score(x_BE_test, y_BE_test))
