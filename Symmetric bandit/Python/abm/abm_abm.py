import numpy as np                                                              # numpy
from abm_structure import abm_structure                                         # Python structure
from abm_task import abm_task                                                   # task  module
from abm_record import abm_record                                               # record module
from abm_construct import abm_construct                                         # agent selection module

def abm_abm(P):
    
    """
    This function runs the symmetric Bandit task in generative or evaluative 
    mode.

    Input
        P   : parameter structure
    
    Output
        P or log likelihood

    """
    # initialization
    # -------------------------------------------------------------------------
    task            = abm_task(P)                                               # task block model initialization
    P.task          = task                                                      # parameter structure extension  
    agent           = abm_construct(P.agent, P)                                 # agent initialization
    record          = abm_record(P)                                             # record initialization    
    agent.start_task()                                                          # task start    
    
    # block iterations
    # -------------------------------------------------------------------------
    for k in np.arange(task.K):                                                 # block iterations
        agent.start_block()                                                     # block start
        task.k      = k                                                         # task block counter
        agent.k     = task.k                                                    # agent block counter
    
        # trial iterations
        for t in np.arange(task.T):                                             # trial iterations
            task.t      = t                                                     # task trial counter
            agent.t     = task.t                                                # agent trial counter                       
            task.state()                                                        # task state realization
            agent.action()                                                      # agent action realization
            task.a      = agent.a                                               # agent action
            task.reward()                                                       # task reward 
            agent.r     = task.r                                                # task reward  
            agent.learn()                                                       # agent reward processing
            record.rec(task, agent)                                             # record model variables    
    
        # end block
        # ---------------------------------------------------------------------
        record.save()                                                           # block data saving

    # end task
    # -------------------------------------------------------------------------
    agent.end_task ()                                                           # agent task end

    # return
    # -------------------------------------------------------------------------
    if P.mode == "gen":                                                         # generative mode
        return P                                                                # no return
    elif P.mode == "llh":                                                       # log likelihood mode
        return record.llh                                                       # log likelihood     