# import the library
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")        # import the dataset
x = dataset.iloc[:, :-1].values             # extract the independent variable
y = dataset.iloc[:, 3]                      # extract the dependent variable

""" handling missing data (replace missing data with mean value) """

imputer = SimpleImputer(strategy='mean')
# fit imputer object to the independent variables x
imputer = imputer.fit(x[:, 1:3])
# Replace missing data with the calculated mean value
x[:, 1:3] = imputer.transform(x[:, 1:3])

""" Categorical data for country variable """

label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# Encode for dummy variables
onehot_encoder = ColumnTransformer(
    [("Country", OneHotEncoder(), [0])], remainder="passthrough")
x = onehot_encoder.fit_transform(x)

# encoding for purchased variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

"""Split the dataset into train and test set"""

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

"""Feature scaling of datasets"""

st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
