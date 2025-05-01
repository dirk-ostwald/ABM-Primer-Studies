import scipy.stats as rv                                                        # random variable module
import numpy as np                                                              # numpy


def abm_llh(theta, a_t, c_t):

    """"
    This function evaluates the analytical form of the Gabor task agent-based"
    behavior model (ABM)"
    
    Input
        theta   : parameter vector (sigma, tau)
        a_t     : actions
        c_t     : contrast differences
  
    Output
        log likelihood value
    """        
    T1 = rv.bernoulli.pmf(a_t, theta[1])*rv.norm.cdf(0, c_t, theta[0])
    T2 = rv.bernoulli.pmf(a_t, 1-theta[1])*(1 - rv.norm.cdf(0, c_t, theta[0]))
    return np.sum(np.log(T1 + T2))

    

