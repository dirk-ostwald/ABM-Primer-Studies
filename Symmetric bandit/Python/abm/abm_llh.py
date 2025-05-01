import numpy as np
import pandas as pd
import scipy.stats as rv

# Initial distributions
# -----------------------------------------------------------------------------
def p_s1(s):

    """
    Probability density function for s_1
    """

    return(rv.uniform.rvs(s, 0, 1))

def r_s1():

    """
    Random sample from p(s_1)
    """

    return(rv.uniform.rvs(0, 1))

def p_d1(d):

    """
    Probability density function for s_1
    """

    return(int(d == 0))

def r_d1():

    """
    Random sample from p(s_1)
    """

    return(0)

def p_alpha_1(alpha):

    """
    Probability mass function for alpha_1
    """

    return(int(np.array_equal(alpha, np.array([1,1]))))

def r_alpha_1():

    """
    Random sample from p(alpha_1)
    """

    return(np.array([1,1]))

# transition distributions
# -----------------------------------------------------------------------------
def p_ot_g_st(o, s):

    """
    Probability mass function of o_t given s_t
    """
    return(rv.bernoulli.pmf(o, s))

def r_ot_g_st(s):

    """
    Random sample from p(o_t|s_t)
    """
    return(rv.bernoulli.rvs(s))

def p_at_g_dt(a,d,tau):

    """
    Probability mass function of a_t given d_t
    """
    return(rv.bernoulli.pmf(a,tau)**d * (rv.bernoulli.pmf(a,1-tau))**(1-d)) 

def r_at_g_dt(d,tau):

    """
    Random sample from p^tau(a_t|d_t)
    """
    return(rv.bernoulli.rvs(tau)**d * (rv.bernoulli.rvs(1-tau))**(1-d)) 

def p_st_g_stm1(s_t, s_tm1):

    """
    Probability mass function of s_t given s_tm1
    """
    return(int(s_t == s_tm1))

def r_st_g_stm1(s_t):

    """
    Random sample from p(s_t|s_{t-1})
    """
    return(s_t)


def p_bt_g_btm1_ot(b_t, b_tm1, o_t):

    """
    Probability mass function of b_t given b_{t-1} and o_t
    """
    return(int(np.array_equal(b_t, b_tm1 + np.array([1-o_t,o_t]))))


def r_bt_g_btm1_ot(b_tm1, o_t):

    """
    Random sample from p(\b_t|\b_{t-1},o_t)
    """
    return(b_tm1 + np.array([1-o_t[0],o_t[0]]))


def p_dt_g_bt(d_t, b_t):

    """
    Probability mass function of d_t given b_t
    """
    return(int((b_t[0] >= b_t[1])**(1-d_t) * (b_t[0] < b_t[1])**d_t))

def r_dt_g_bt(b_t):

    """
    Random sample from p(d_t|b_t)
    """
    return(int(b_t[0] < b_t[1]))

# simulation
# -----------------------------------------------------------------------------

T       = 10
df      = pd.DataFrame(np.nan, index=range(T), columns = ["t", "s_t", "o_t", "b_t", "d_t", "a_t"])
tau     = 1  
T       = 31     
t       = np.full((T, 1), np.nan)
s_t     = np.full((T, 1), np.nan)
o_t     = np.full((T, 1), np.nan)
b_t     = np.full((T, 2), np.nan)
d_t     = np.full((T, 1), np.nan)
a_t     = np.full((T, 1), np.nan)


for i in range(T):

    if i == 0:
        t[i]     = 1
        s_t[i,:] = r_s1()                                                       # initial state
        o_t[i,:] = r_ot_g_st(s_t[i])                                            # initial observation
        b_t[i,:] = r_alpha_1()                                                  # initial alpha
        d_t[i,:] = r_d1()                                                       # initial decision
        a_t[i,:] = r_at_g_dt(d_t[i], tau)                                       # initial action
    else:
        t[i]    = i + 1
        s_t[i,:] = r_st_g_stm1(s_t[i-1])                                        # state transition     
        o_t[i,:] = r_ot_g_st(s_t[i])                                            # observation
        b_t[i,:] = r_bt_g_btm1_ot(b_t[i-1,:], o_t[i])                           # belief state sufficient statistics              
        d_t[i,:] = r_dt_g_bt(b_t[i,:])                                          # decision                
        a_t[i,:] = r_at_g_dt(d_t[i], tau)                                       # action

df = pd.DataFrame(np.hstack([t, s_t, o_t, b_t, d_t, a_t]),                       # stack arrays horizontally
                  columns  =["t", "s_t", "o_t", "b_t_1", "b_t_2", "d_t", "a_t"]) # assign column names

print(df)