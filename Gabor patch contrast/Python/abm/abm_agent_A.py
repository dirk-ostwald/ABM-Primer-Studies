import numpy as np                                                              # numpy
import scipy.stats as rv                                                        # random variable module 

class abm_agent:

    # agent instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):
        
        """
        This function is the instantiation operation of the agent A class

        Input
            self    : agent object
            P       :  parameter structure
        
        Output
            initialized agent object
        """

        # agent attributes
        # ---------------------------------------------------------------------      
        # mode distinction
        self.mode   = P.mode                                                    # mode
        if self.mode == "gen":                                                  # generative mode
            self.n_s    = 1                                                     # number of observation samples     
        elif self.mode == "llh":                                                # log likelihood mode
            self.df     = P.df                                                  # data frame
            self.n_s    = P.n_s                                                 # number of observation samples
            self.pa     = np.array([np.nan])                                    # observed action probability mass

        # parameter
        self.sigma  = P.sigma                                                   # fluctuation standard deviation
        
        # structural components
        self.task   = P.task                                                    # task
        self.T      = P.T                                                       # number of trials per block
        self.O      = np.array(np.linspace(-3,3,int(1e3)))                      # observation set subset 
        self.B      = np.array([0,1])                                           # belief state support
        self.D      = np.array([0,1])                                           # decision space

        # dynamic components
        self.k      = np.array([np.nan])                                        # block counter
        self.t      = np.array([np.nan])                                        # trial counter
        self.c      = np.full((self.n_s,1), np.nan)                             # contrast difference
        self.o      = np.full((self.n_s,1), np.nan)                             # observation
        self.b      = np.full((self.n_s,2), np.nan)                             # belief state    
        self.v      = np.full((self.n_s,2), np.nan)                             # decision value 
        self.d      = np.full((self.n_s,2), np.nan)                             # decision

        # agent-based behavioral model attributes
        # ---------------------------------------------------------------------   
        self.tau    = P.tau                                                     # action noise
        self.a      = np.array([np.nan])                                        # action
        self.rt     = np.array([np.nan])                                        # response time                     
    
    # agent functions
    # -------------------------------------------------------------------------
    def p_o_giv_c(self, o):
        
        """
        This function evaluates the observation distribution p(o_t|c_t).
        
        Inputs
                self    : agent object
                
        Output
                observation probability density     
         
        """
        return rv.norm.pdf(o, self.c, self.sigma)
        
    def r_o_giv_c(self):
        
        """ 
        This function samples the observation distribution p(o_t|c_t).
                               
        Inputs
                self    : agent object
        
        Outputs
                self    : agent object with updated attribute
                    .o  : observation  
        """
        self.o = rv.norm.rvs(self.c, self.sigma, size = self.n_s)     
                                                        

    def beta(self):

        """ 
        This function evaluates the the agent belief state function \beta.
        
        Input
            self    : agent object
        
        Output
            self    : agent object with updated attribute
                .b  : belief state function parameter 
        """
        # parameter values
        kappa       = self.task.kappa
        sigma       = self.sigma
        
        # belief state function constituents
        u           = rv.norm.cdf(     0, self.o, sigma)
        v           = rv.norm.cdf(-kappa, self.o, sigma)
        w           = rv.norm.cdf( kappa, self.o, sigma)
                
        # belief state evaluation
        self.b[:,0]   = (u - v) / (w - v)     
        self.b[:,1]   = 1 - self.b[:,0]

 
    def phi(self):
 
        """ 
        This function implements the agent's decision valence function \phi.
         
        Input
            self    : agent object
            
        Output
            self    : agent object with updated attribute
                .v  :  decision valence function values
        
        """
        for d in self.D:
            self.v[self.b[:,d] >= 0.5, d] = 1                                   # v_t^d = 1 if b_t^d >= 0.5
            self.v[self.b[:,d]  < 0.5, d] = 0                                   # v_t^d = 0 if b_t^d < 0.5   
       

    def delta(self):
        
        """
        This function implements the agent's decision function \delta.
        
        Input
            self    : agent object
            
        Output
            self    : agent object with updated attribute
               .d   : decision
        
        
        """
        # np.argmax note: in case of multiple occurrences of the maximum values, 
        # the indices corresponding to the first occurrence are returned.
        self.d  = self.D[np.argmax(self.v, axis = 1)]
     
          

    def learn(self):
        
        """
        This function implements the agent's learning function.
        """
        pass     
    
    # agent-based behavioral model functions
    # -------------------------------------------------------------------------
    def p_a(self):
        
        """
        This function evaluates the Monte-Carlo estimate of  p^\tau(a_t|d_t).
        
        Inputs
            self    : agent object
            a       : action
        
        Output
            action probability mass
        
        """
        # \hat{p}^{tau}(a_t|d_t)      
        T1          = rv.bernoulli.pmf(self.a,self.tau)*(1-np.mean(self.d))
        T2          = rv.bernoulli.pmf(self.a,1-self.tau)*np.mean(self.d)
        self.pa     = T1 + T2
    
    def r_a(self):
        
        """
        This function samples the action distribution p(a_t|d_t).
        
        Input
            self    : agent object
        
        Output
            self    : agent object with updated attribute
                .a  : action
        """
        # p^{tau}(a_t|d_t)
        self.a = ((rv.bernoulli.rvs(1-self.tau))**self.d)*\
                  (rv.bernoulli.rvs(self.tau))**(1-self.d)   
   
    def action(self):
        """
        This function implements the agent-based behavioral model's action.
        
        Input
            self    : agent object      
          
        Output
            self    : agent object with updated attribute
                .a      : action (generative mode)
                .p_a    : action probability (log likelihood mode)
        """
        self.r_o_giv_c()                                                        # observation realization
        self.beta()                                                             # belief state evauluation
        self.phi()                                                              # decision valence evaluation
        self.delta()                                                            # decision evaluation
        if self.mode == "gen":                                                  # generative mode
            self.r_a()                                                          # action realization
        elif self.mode == "llh":                                                # log likelihood mode
            df, k, t    = self.df, self.k, self.t                               # data frame, block, trial
            self.a      = df.a_t[df.k == k][df.t == t].values[0]                # action observation        
            self.p_a()                                                          # action probability evaluation

    # auxiliary functions
    # -------------------------------------------------------------------------
    def start_task(self):
        pass
    
    def start_block(self):
        pass

    def end_task(self):
        pass
