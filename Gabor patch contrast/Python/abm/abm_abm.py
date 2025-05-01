import numpy as np                                                              # numpy
from abm_structure import abm_structure                                         # Python structure
from abm_task import abm_task                                                   # task  module
from abm_record import abm_record                                               # record module
from abm_construct import abm_construct                                         # agent selection module

def abm_abm(P):

    """
    This function runs the Gabor contrast task in generative or evaluative mode.

    Input
        P   : parameter structure
    
    Output
        P or log likelihood

    """
    # initialization
    # -------------------------------------------------------------------------
    task            = abm_task(P)                                               # task block model initialization
    P.task          = task                                                      # agent initialization structure    
    agent           = abm_construct(P.agent, P)                                 # agent initialization
    record          = abm_record(P)                                             # record initialization    
    agent.start_task()                                                          # agent task start  
   
    # block iterations
    # -------------------------------------------------------------------------
    for k in np.arange(task.K):                                                 # block iterations
         agent.start_block()                                                    # agent block start  
         task.k  = k                                                            # block counter
         agent.k = k                                                            # block counter   

         # trial iterations
         # ---------------------------------------------------------------------
         for t in np.arange(task.T):                                            # trial iterations
            task.t  = t                                                         # trial counter              
            agent.t = task.t                                                    # trial counter
            task.state()                                                        # task state realization             
            agent.c = task.c                                                    # agent sensory inputs
            agent.action()                                                      # agent action
            task.a  = agent.a                                                   # task action information            
            task.reward()                                                       # task reward realization       
            agent.r = task.r                                                    # agent reward information
            agent.learn()                                                       # agent reward processing
            record.rec(task, agent)                                             # record model variables        

         # end block
         # --------------------------------------------------------------------
         record.save()                                                           # save data each block 
    
    # end task
    # -------------------------------------------------------------------------
    agent.end_task ()                                                           # agent task end
        
    
    # return
    # -------------------------------------------------------------------------
    if P.mode == "gen":                                                         # generative mode
        return P                                                                # no return
    elif P.mode == "llh":                                                       # log likelihood mode
        return record.llh                                                       # log likelihood                                        
