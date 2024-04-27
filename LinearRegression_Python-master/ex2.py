

import pandas as pd 

pathtodata = 'Exercise_Data/ex2_Data.txt'

data = pd.read_csv(pathtodata,delimiter = ',',header=None)

x = data[[0,1]]
x.columns = ['X1','X2']

y = data[[2]] # Price of houses in Port-land

m = len(y)

import featureNormalize as fn

[x_norm, x_mean, x_std] = fn.featureNormalize(x)

# Please check the MLplot source code for details
import MLplot as pl

dataPlot = pl.plot2D(x_norm,y)
dataPlot.set_title("Profit and Population")
dataPlot.set_xlabel('Size of the house and (X1) and number of bedrooms (X2)')
dataPlot.set_ylabel('Price of the house ($)')

import numpy as np
import gradientDescent as gd

x_norm.insert(loc=0, column='X0', value=np.ones(m))

num_of_feat = len(x_norm.columns)
num_of_train = len(x_norm)

alpha = 0.1
num_iters = 400
theta_0 = pd.DataFrame(np.zeros([num_of_feat,1]))

[theta,Jhist,thetahist] = gd.gradientDescent(x_norm,y,theta_0,alpha,num_iters)


iterations = pd.DataFrame(list(range(num_iters)))
iterations.columns = ['Iter']

Leacur_plot = pl.plot2D(iterations,Jhist)
Leacur_plot.set_title(r'Learning curve for $\alpha$={0}'.format(alpha))
Leacur_plot.set_xlabel('Iterations')
Leacur_plot.set_ylabel('Cost Function')

reg_plot = pl.regressionPlot(x_norm,y,theta,1)
reg_plot.set_title("Linear Regression")
reg_plot.set_xlabel('Size of the house normalized')
reg_plot.set_ylabel('Price of the house ($)')
