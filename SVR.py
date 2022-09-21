# SVR

# Data Preprocessing
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the SVR to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

# Predicting the test set results
y_predict = regressor.predict(x)
