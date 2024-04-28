
GRADIENTDESCENT Processa o gradiente para aprender theta
theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) atualiza o valor de theta ao usar num_iters com a razão de aprendizado alpha

def gradientDescent(x, y, theta, alpha, num_iters):
    
    import numpy as np
    import pandas as pd
    import computeCost as cc
    
    m = len(y)
    
    iterations = list(range(num_iters))
    
    term = pd.DataFrame(np.zeros([m,num_of_feat]))
    
    Jhist = pd.DataFrame(iterations)
    
    thetahist = pd.DataFrame(np.zeros([num_iters,num_of_feat]))
    
    for i in range(num_iters):

        gen_term = np.dot(x,theta)-y
        
        for d in range(num_of_feat):
            
            term.loc[:,d] = gen_term.mul(x.iloc[:,d],axis=0)
        
            theta.iloc[d,:] =  theta.iloc[d,:] - ((alpha/m) * np.sum(term.iloc[:,d]))
        
            thetahist.iloc[i,d] = float(theta.iloc[d,:])
        
        Jhist.iloc[i,0] = float(cc.computeCost(x,y,theta))
        
    theta.columns = ["Theta"]
    Jhist.columns = ["Jhist"]
    
    return [theta,Jhist,thetahist]
