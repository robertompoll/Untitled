def computeCost(x, y, theta):
    
    import numpy as np
    
    m = len(y)
    
    J = 0
    
    hip = np.dot(x,theta) 
    
    squaErr = np.power((hip - y),2)
    
    J =  (1 / (2 * m)) * (np.sum(squaErr))
    
    J.columns = ["J"]
    
    return J
