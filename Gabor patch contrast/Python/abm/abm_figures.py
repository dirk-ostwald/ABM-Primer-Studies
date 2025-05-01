import numpy as np                                                              # NumPy

def abm_figures(plt):

    """ 
    This function sets some plt defaults and returns blue and red color 
    palettes
    
        Input
            plt     : Matplotlib instance
        
        Output
            plt     : update Matplotlib instance
            colors  : blue and red color palettes
    
    """

    # plt default parameters
    rcParams =  {   'font.family'           : 'serif',  
                    'font.serif'            : 'Computer Modern Serif',
                    'mathtext.fontset'      : 'cm', 
                    'text.usetex'           : 'True', 
                    'axes.spines.top'       : 'False',
                    'axes.spines.right'     : 'False',
               }   
    plt.rcParams.update(rcParams)
    
    # color palettes
    blues = np.array([[7,47,95],[18,97,160],[ 56,149,211],[ 88,204,237]])/255   # shades of blue
    reds  = np.array([[167,0,0],[255,82,82],[255,123,123],[255,186,186]])/255   # shades of reds
    
    return plt, blues, reds 
