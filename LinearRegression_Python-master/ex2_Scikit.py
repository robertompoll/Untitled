
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import MLplot as pl

pathtodata = 'Exercise_Data/ex2_Data.txt'

data = pd.read_csv(pathtodata,delimiter = ',',header=None)

x = data[[0,1]] # Size of the house and (X0) and number of bedrooms (X1)
y = data[[2]] # Price of houses in Port-land

x_norm = pd.DataFrame(preprocessing.scale(x))

dataPlot = pl.plot2D(x_norm,y)
dataPlot.set_title("Profit and Population")
dataPlot.set_xlabel('Size of the house and (X1) and number of bedrooms (X2)')
dataPlot.set_ylabel('Price of the house ($)')

m = len(y)

[X_train, X_test, y_train, y_test] = train_test_split(x_norm,y,test_size=0.3,
random_state=101)

lm = LinearRegression()

lm.fit(X_train,y_train)

theta0 = pd.DataFrame(lm.intercept_)
theta = pd.DataFrame(lm.coef_)

x_norm.insert(loc=0, column="X0", value=np.ones(m))
theta.insert(loc=0, column="t0", value=theta0)

]x_norm.columns = ['x0','x1','x2']

# Choose the header feature index
theta.columns = ['t0','t1','t2']

# Transpose theta (for pl.regressionPlot)
theta = theta.transpose()

# Plotting the linear Regression
reg_plot = pl.regressionPlot(x_norm,y,theta,1)
reg_plot.set_title("Linear Regression")
reg_plot.set_xlabel('Size of the house normalized')
reg_plot.set_ylabel('Price of the house ($)')
