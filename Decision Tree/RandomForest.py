# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# load the dataset
dataset = pd.read_csv("dataset/Social_Network_Ads.csv")

# extract the independent and dependent variable
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# Feature scaling
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

R_forest = RandomForestClassifier(n_estimators=10, criterion="entropy")
R_forest.fit(x_train, y_train)

y_pred = R_forest.predict(x_test)  # Predict the test results
cm = confusion_matrix(y_test, y_pred)  # create the confusion matrix

# visualize the train results
x1, x2 = np.meshgrid(np.arange(start=x_train[:, 0].min() - 1, stop=x_train[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_train[:, 1].min() - 1, stop=x_train[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, R_forest.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
    x1.shape), alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train == j, 0], x_train[y_train == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Naive Bayes (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualize the test results
x1, x2 = np.meshgrid(np.arange(start=x_test[:, 0].min() - 1, stop=x_test[:, 1].max() + 1, step=0.01),
                     np.arange(start=x_test[:, 1].min() - 1, stop=x_test[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, R_forest.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
    x1.shape), alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1],
                c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Naive Bayes (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
