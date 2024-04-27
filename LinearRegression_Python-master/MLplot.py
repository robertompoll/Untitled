

def plotstyle():
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(5,4),dpi=80) 

    axes = fig.add_axes([0,0,1,1])

    axes.set_title("2D plot")
    axes.set_xlabel("x values")
    axes.set_ylabel("y values")
    
    return [fig , axes]

def plot2D(x,y):
    
    import numpy as np
    import pandas as pd

    [fig,axes] = plotstyle()
    
    x = pd.DataFrame(x)
    
    head = list(x.columns)
        
    num_of_feat = len(x.columns)
        
    for i in range(num_of_feat):
        
        axes.plot(x.iloc[:,i],y, label= head[i],
                  color=(np.random.sample(),np.random.sample(),
                  np.random.sample()),linewidth=0, linestyle='-',alpha=1,
                  marker='+', markersize=5, markeredgewidth=1)
                  
        axes.legend(loc=0)
    
    return axes

def plot3D(x,y,z):
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
              
    fig = plt.figure(figsize=(8,7)) 
    axes = fig.gca(projection='3d')
    
    axes.plot_surface(x,y,z,linewidth=1, cmap=cm.coolwarm)
       
    axes.set_xlabel('Parameter')
    axes.set_ylabel('Parameter')
    axes.set_zlabel('Parameter')
    
    
    fig.savefig('3Dplot.png') # - Save the figure
    # axes.set_xlabel(r'$\theta_0$') # - Using special characters
    # axes.view_init(60, 35) # - Chenge the view of the plot
    # ax.set_zlim(0, 800) # - Set the z range axis value
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) # - set the number style
    # fig.colorbar(surf, shrink=0.5, aspect=5) # Adding a colorbar legend
    
    
    return axes


def regressionPlot(x,y,theta,refcol):
    
    import numpy as np
    import pandas as pd

    [fig , axes] = plotstyle()
    
    x = pd.DataFrame(x)

    head = list(x.columns)
        
    x_ref = x.iloc[:,[0,refcol]]
            
    axes.plot(x.iloc[:,refcol],y, label= head[refcol],
                  color=(np.random.sample(),np.random.sample(),
                  np.random.sample()),linewidth=0, linestyle='-',alpha=1,
                  marker='+', markersize=5, markeredgewidth=1)
    
    reg = np.dot(x_ref,theta.iloc[[0,refcol],:])
    
    axes.plot(x.iloc[:,refcol],reg,label='Regression',color='blue',linewidth=1)

    axes.set_title('Regression plot')
    axes.set_xlabel('Parameter')
    axes.set_ylabel('Parameter')
   
    axes.legend(loc=0)
    
    # fig.savefig('RegressionPlot.png')
    
    return axes

