import numpy as np                                                              # numpy
import scipy.stats as rv                                                        # random variable module

class abm_agent:

    # agent instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):
        
        """
        This function is the instantiation operation of the agent A class

        Input
            P   :  parameter structure
        
        Output
           initialized agent object
        
        """

        # agent attributes
        # ---------------------------------------------------------------------
        # mode
        self.mode   = P.mode                                                    # task mode (generative or log likelihood)

        # data frame
        if self.mode == "llh":                                                  # log likelihood mode
            self.df = P.df                                                      # data frame
        
        # parameters
        self.tau    = P.tau                                                     # observation noise
    
        # structural components
        self.S      = P.task.S                                                  # continuous state space [0,1]
        self.D      = np.array([0,1])                                           # discrete decision space {0 (left button), 1 (right button)}
            
        # dynamic components variables initialization
        self.k      = np.nan                                                    # block counter
        self.t      = np.nan                                                    # trial counter
        self.v      = np.array([np.nan, np.nan])                                # decision value 
        self.d      = np.nan                                                    # decision         
        self.r      = np.nan                                                    # reward
        self.rr     = 0                                                         # cumulative reward
        self.rt     = np.nan                                                    # response time
   
    # agent functions
    # ------------------------------------------------------------------------- 
    def beta(self):

        """
        This function implements the agent's belief state function \beta.

        Input
            self    : agent object 

        Output
            self    : agent object with updated field 
                .alpha 

        """
        a,r         = self.a, self.r                                            # observations
        dalpha      = np.array([(r^a)*((1-r)^(1-a)), ((1-r)^a)*((r)^(1-a))])    # belief state update
        self.alpha  = self.alpha + dalpha                                       # belief state parameter
    
    def phi(self):
    
        """
        This function implements the agent's decision value function \phi
        
        Input
            self    : agent object
            
        Output
            self    : agent object with updated field 
                .v
        """
        E       = self.alpha[0]/(self.alpha[0] + self.alpha[1])                 # belief state expectation
        self.v  = np.array([E,1-E])                                             # decision value array
   
    def delta(self):

        """
        This function implements the agent's decision function \delta.

        Input
            self    : agent object 
        
        Output  
            self    : agent object with updated field 
            .d

        """
        # np.argmax note: in case of multiple occurrences of the maximum values, 
        # the indices corresponding to the first occurrence are returned.
        self.d  = self.D[np.argmax(self.v)]                                     # optimal decision

    def action(self):
        """
        This function implements the agent-based behavioral model's action.
        
        Input
            self    : agent object      
          
        Output
            self    : agent object with updated attribute
                .a  : action
        """
        self.phi()                                                              # decision value evaluation
        self.delta()                                                            # decision evaluation
        if self.mode == "gen":                                                  # generative mode
            self.r_a()                                                          # action realization                                

        elif self.mode == "llh":                                                # log likelihood mode
            df, k, t    = self.df, self.k, self.t                               # data frame, block, trial
            self.a      = df.a_t[df.k == k][df.t == t].values[0]                # action observation
            self.p_a_giv_d()                                                    # action probability evaluation
        

    def learn(self):
        
        """
        This function implements the agent's learning function.
        """
        self.beta()                                                             # belief state update

    # agent-based behavioral model functions
    # -------------------------------------------------------------------------
    def p_a_giv_d(self):
        
        """
        This function evaluates the action distribution p(a_t|d_t).
        
        Inputs
            self    : agent object
         
        Output
            action probability mass
        
        """
        # p^{tau}(a_t|d_t)
        self.p_ad = ((rv.bernoulli.pmf(self.a, 1-self.tau))**self.d)*\
                    ((rv.bernoulli.pmf(self.a, self.tau))**(1-self.d))                 
    
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
   
   
    # auxiliary functions
    # -------------------------------------------------------------------------
    def start_task(self):
        pass
    
    def start_block(self):
        self.alpha  = np.array([1,1])                                           # initial belief state parameter \alpha_0

    def end_task(self):
        pass