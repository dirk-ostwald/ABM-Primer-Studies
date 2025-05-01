import numpy as np                                                              # numpy
import scipy.stats as rv                                                        # random variable module 

class abm_task:

    # task instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):
        
        """
        This function is the instantiation operation of the task class.

        Input
            P       : parameter structure
        
        Output
            Initialized task object
        
        """

        # task attributes
        # ---------------------------------------------------------------------
        self.mode   = P.mode                                                    # mode
        if self.mode == "llh":                                                  # log likelihood mode
            self.df = P.df                                                      # data frame
           
        # parameters
        self.P      = P.P                                                        # participant number
        self.K      = P.K                                                       # number of blocks
        self.T      = P.T                                                       # number of trials per block
        
        # structural components
        self.S      = np.array([0,1])                                           # discrete state space {0 (higher contrast left), 1 (higher contrast right)}
        self.kappa  = 1                                                         # maximal nomralized absolute contrast difference parameter
        self.C      = np.linspace(-self.kappa,self.kappa, int(1e3))             # contrast difference space
        self.A      = np.array([0,1])                                           # discrete action space {0 (left button press), 1 (right button press)}
        self.R      = np.array([0,1])                                           # discrete reward space {0,1}

        # dynamic components
        self.k      = np.array([np.nan])                                        # block counter
        self.t      = np.array([np.nan])                                        # trial counter
        self.s      = np.array([np.nan])                                        # state
        self.c      = np.array([np.nan])                                        # contrast difference
        self.a      = np.array([np.nan])                                        # action
        self.r      = np.array([np.nan])                                        # reward

    # task functions
    # -------------------------------------------------------------------------
    def p_s(self, s):
        
        """
        This function evaluates the state distribution p(s_t).
        
        Input 
                self    : task object
                s       : state 
                
        Output
               state value probability mass     
         
        """
        return rv.bernoulli.pmf(s, 0.5)
        
    def r_s(self):
        
        """ 
        This function samples the state distribution p(s_t).
                               
        Input 
                self    : task object
        
        Outputs
                self    : task object with updated attribute
                    .s  : state  
        """
        self.s = rv.bernoulli.rvs(0.5)  

    def p_c_s(self,c,s):
                     
        """
        This function evaluates the contrast distribution p(c_t|s_t).
        
        Inputs
                self    : task object
                s       : state  
                c       : contrast
                
        Output
               contrast value probability density     
         
        """
        # note that scipy.stats.uniform takes parameters a,b for U(x;[a,a+b])
        return (rv.uniform.pdf(c, -self.kappa, self.kappa)**(1-s[0])) *\
               (rv.uniform.pdf(c, 0          , self.kappa)**s[0])

 
    def r_c_s(self):
        
        """ 
        This function samples the contrast distribution p(c_t|s_t) 
                self    : task object
        
        Outputs
                self    : task object with updated attribute
                    .c  : state  
        """  
        self.c = (rv.uniform.rvs(-self.kappa, self.kappa)**(1-self.s)) *\
                 (rv.uniform.rvs(0          , self.kappa)**self.s)
       

    def state(self):

        """
        This function samples the unobserved state and observed contrast

        Input s
                self    : task object
        
        Output
                self    : task object with updated attributes
                    .s
                    .c
        """
        if self.mode == "gen":                                                  # sampling for experimental and generative modes
            self.r_s()                                                          # state realization    
            self.r_c_s()                                                        # contrast realization
        elif self.mode == "llh":                                                # evaluation for log likelihood mode                      
            df, k, t    = self.df, self.k, self.t                               # data frame, block, trial
            self.c      = df.c_t[df.k == k][df.t == t].values[0]                # contrast difference  observation       
    
    def p_r_sa(self,r,s,a):

        """
        This function evaluates the reward distribution p(r_t|s_t,a_t).
        
        Inputs
                self    : task object
                r	: reward
		s       : state  
                a       : action  
                
        Output
               reward value probability mass     
        """
        return rv.bernoulli.pmf(r, int(s == a))  
    
    def r_r_sa(self):

        """
        This function samples the reward distribution p(r_t|s_t,a_t).
        
        Inputs
                self    : task object
                
        Outputs
                self    : task object with updated attribute
                    .r  : reward 
        """
        self.r =  rv.bernoulli.rvs(int(self.s == self.a))  

    def reward(self):

        """
        This function samples the reward.

        Input s
                self    : task object
        
        Output
                self    : task object with updated attributes
                    .r
        """
        self.r_r_sa()                                                                    