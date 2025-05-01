from psychopy import visual, event, core
import numpy as np
import os

class abm_agent:

    # agent instantiation
    # -------------------------------------------------------------------------
    def __init__(self, a_init):
        
        """
        This function is the instantiation operation of the agent A class

        Input
            a_init :    agent initialization structure
        """

        # agent attributes
        # ---------------------------------------------------------------------

        # Visual presentation parameters
        self.winsize    = [2560, 1440]                                          # PsychoPy window size    
        self.fs         = 0.2                                                   # relative fontsize                                 

        # behavior acquisition parameters
        self.keys       = ['left', 'right']                                     # keys (actions)
        self.key2act    = {'left' : 0, 'right' : 1}                             # key encoding 
        self.ltime      = 1                                                     # reward learning time

        # PsychoPy startup
        self.win        = visual.Window(self.winsize, color = [0, 0, 0])        # grey background

        # dynamic components variables initialization
        self.k          = np.nan                                                # block counter
        self.alpha      = np.array([np.nan, np.nan])                            # belief state
        self.v          = np.array([np.nan, np.nan])                            # decision value 
        self.d          = np.nan                                                # decision         
        self.r          = np.nan                                                # reward
        self.rt         = np.nan                                                # response time

    # agent functions
    # -------------------------------------------------------------------------
    def start_task(self):
        
        """
        Task start display until space bar pressed
        """
        
        info    = "Press space bar to start the task"                           # task start information
        text    = visual.TextStim(self.win, text = info)                        # text 
        text.draw()                                                             # draw text on window
        self.win.flip()                                                         # window display
        event.waitKeys(keyList=["space"])                                       # wait for space bar press
    
    def start_block(self):
        
        """
        Block start display until space bar pressed
        """
        info    = "Press space bar to start a new block"                        # block start information           
        text    = visual.TextStim(self.win, text = info)                        # text 
        text.draw()                                                             # draw text on window
        self.win.flip()                                                         # window display
        event.waitKeys(keyList=["space"])                                       # wait for space bar press

    def action(self):

        """
        Agent action collection 
        """
        text = visual.TextStim(self.win, text = "+",  height= self.fs)          # fixation cross
        text.draw()                                                             # draw text on window
        self.win.flip()                                                         # window display
        onset   = core.getTime()                                                # stimulus onset time    
        resp    = np.array([np.nan, np.nan])                                    # initialize response variable
        while resp[0] not in self.keys:                                         # wait for valid key press response
            resp = event.waitKeys(keyList = self.keys, timeStamped=True)[0]     # Wait for key press
        key, t  = resp                                                          # action and key press time step
        self.a  = self.key2act[key]                                             # decision 
        self.rt = t - onset                                                     # response time

    def learn(self):
        
        """
        Reward information display
        """
        text = visual.TextStim(self.win, text = f"{self.r}", height = self.fs)  # reward
        text.draw()                                                             # draw text on window
        self.win.flip()                                                         # window display                                  
        core.wait(self.ltime)                                                   # wait for reward display time
          

    def end_task(self):

        """
        Experiment finalization
        """
        text = visual.TextStim(self.win, text="Task completed")
        text.draw()
        self.win.flip()
        core.wait(2.0)
        self.win.close()


