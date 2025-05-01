import pandas as pd                                                             # Pandas
import numpy as np                                                              # Numpy

class abm_record:
    
    # record instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):
        
        """
        This function is the instantiation operation of the record class.

        Input
            P       : parameter structure
        
        Output
            initialized record object
        
        """  
        # record attributes
        # ---------------------------------------------------------------------
        if P.mode == "gen":                                                     # generative mode
            self.mode   = "gen"                                                 # mode 
            self.df     = pd.DataFrame()                                        # data frame initialization
            self.file   = P.file                                                # filename    
        
        if P.mode == "llh":                                                     # log likelihood mode
            self.mode   = "llh"                                                 # mode
            self.llh    = 0                                                     # log likelihood initialization
        

    # record functions
    # -------------------------------------------------------------------------
    def rec(self, task, agent):

        """
        This function records task and agent states to the record dataframe

        Inputs
            task, agent :   task and agent models
        """
        if self.mode == "gen":                                                  # generative mode 
        
            # variable concatenation
            df_t    = pd.DataFrame({"P"         : [task.P],
                                    "k"         : [task.k],
                                    "t"         : [task.t],
                                    "s_t"       : [task.s ],                       
                                    "alpha_t_1" : [agent.alpha[0]],
                                    "alpha_t_2" : [agent.alpha[1]],
                                    "v_t_1"     : [agent.v[0]],
                                    "v_t_2"     : [agent.v[1]],
                                    "d_t"       : [agent.d],
                                    "a_t"       : [agent.a],
                                    "rt_t"      : [agent.rt],  
                                    "r_t"       : [task.r]})

            # dataframe concatentation
            self.df  = pd.concat([self.df, df_t], ignore_index=True)       

        elif self.mode == "llh":                                                # log likelihood mode
            self.llh += np.log(agent.p_ad)                                      # log likelihood update          


    def save(self):
 
        """
        This function saves the record dataframe to disk
       
        """
        if self.mode == "gen":                                                  # generative mode

            # round to two decimal places
            self.df.round(2).to_csv(self.file,                                  # filename 
                                    sep          = '\t',                        # tab separator
                                    float_format = "%+03.2f",                   # two decimal places and +/-
                                    na_rep       = "NaN",                       # NaN as NaNs
                                    index        = False)                       # no row indices