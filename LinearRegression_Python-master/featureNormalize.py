function [X_norm, mu, sigma] = featureNormalize(X)
FEATURENORMALIZE Normaliza os parâmetros de X
FEATURENORMALIZE(X) retorna a versão tratada de X onde o valor de cada parâmetro é 0 e o desvio é 1.

 def featureNormalize(x):

    import pandas as pd
    import numpy as np

    head = list(x.columns)

    num_of_feat = len(x.columns)
    
    num_of_rows = len(x)
    
    x_norm = pd.DataFrame(np.zeros([num_of_rows,num_of_feat]))
    
    x_mean = pd.DataFrame(x.mean())
    x_std = pd.DataFrame(x.std())
   
    for i in range(num_of_feat):

        x_norm.iloc[:,i]  = (x.iloc[:,i]-x_mean.iloc[i,0])/(x_std.iloc[i,0])
    
    x_norm.columns = head
    
    x_mean.columns = ["X_mean"]
    x_std.columns = ["X_std"]
    
    return [x_norm, x_mean, x_std]


