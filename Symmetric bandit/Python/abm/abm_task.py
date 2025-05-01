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
        self.mode   = P.mode                                                    # task mode (generative or log likelihood)
        if self.mode == "llh":                                                  # log likelihood mode
            self.df = P.df                                                      # data frame
       
        # parameters
        self.K      = P.K                                                       # number of blocks
        self.T      = P.T                                                       # number of trials per block
        self.P      = P.P                                                       # participant index

        # structural components
        self.S      = np.linspace(0,1,int(1e3))                                 # continuous state space [0,1]
        self.A      = np.array([0,1])                                           # discrete action space {0 (blue square), 1 (red square)}
        self.R      = np.array([0,1])                                           # discrete reward space {0,1}
    

        # dynamic components variables initialization
        self.k      = np.nan                                                    # block counter
        self.o      = np.nan                                                    # observation
        self.b      = np.array([np.nan, np.nan])                                # belief state    
        self.v      = np.array([np.nan, np.nan])                                # decision value 
        self.d      = np.nan                                                    # decision         
        self.r      = np.nan                                                    # reward
        self.rr     = 0                                                         # cumulative reward
        self.rt     = np.nan                                                    # response time

    # task functions
    # -------------------------------------------------------------------------
    def p_s_1(self, s):

        """
        This function evaluates the initial state distribution p(s_1)

        Input
            self    : task object
            s       : state

        Output
            state probability density
        
        """
        return  rv.uniform.pdf(s)

    def r_s_1(self):

        """
        This function samples the initial state distribution p(s_1)

        Input
            self    : task object

        Output
            self    : task object with updated field
                .s
        
        """
        self.s  = rv.uniform.rvs()

    def state(self):
        
        """
        This function evaluates the task state. State updates are realized 
        virtually by not altering the task state.

        Input
            self    : task object
        
        Output
            self    : task object with updated field
                .s
        """
        # initial state
        if self.t == 0:
            self.r_s_1()                                                        # s_1 realization                                                


    def p_r_sa(self,r,s,a):

        """
        This function evaluates the reward distribution p(r_t|s_t,a_t).
        
        Inputs
                self    : task object
                r       : reward
                s       : state  
                a       : action  
                
        Output
               reward value probability mass     
        """
        return ((rv.bernoulli.pmf(r,s))**a) *\
               ((rv.bernoulli.pmf(r,1-s))**(1-a))  
    
    def r_r_sa(self):

        """
        This function samples the reward distribution p(r_t|s_t,a_t).
        
        Inputs
                self    : task object
                
        Outputs
                self    : task object with updated attribute
                    .r  : reward
        """
        self.r = ((rv.bernoulli.rvs(self.s))**self.a) *\
                 ((rv.bernoulli.rvs(1-self.s))**(1-self.a)) 

    def reward(self):

        """
        This function samples the reward.

        Input s
                self    : task object
        
        Output
                self    : task object with updated attributes
                    .r
        """
        if self.mode == "gen":                                                  # generative mode
            self.r_r_sa()
        
        elif self.mode == "llh":                                                # log likelihood mode
            df, k, t = self.df, self.k, self.t                                  # data frame, block, trial                        
            self.r   = df.r_t[df.k == k][df.t == t].values[0]                   # reward observation


 