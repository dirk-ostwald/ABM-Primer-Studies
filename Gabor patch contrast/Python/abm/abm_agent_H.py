from psychopy import visual, event, core                                        # PsychoPy library for visual stimuli and event handling        
import numpy as np                                                              # numerical operations                                  
from pathlib import Path                                                        # path management


class abm_agent:

    # agent instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):
        
        """
        This function is the instantiation operation of the agent H class

        Input
            self    : agent object
            P       :  parameter structure
        
        Output
            initialized agent object
        """

        # agent attributes
        # ---------------------------------------------------------------------
        # visual presentation parameters
        self.sdir       = P.sdir                                                # stimulus directory
        self.winsize    = [1000, 1000]                                          # PsychoPy window size    
        self.fs         = 0.05                                                  # relative fontsize     

        # behavior acquisition parameters
        self.keys       = ['left', 'right']                                     # keys (actions)
        self.key2act    = {'left' : 0, 'right' : 1}                             # key encoding 
        self.ltime      = 1                                                     # reward learning time

        # PsychoPy startup
        self.win        = visual.Window(self.winsize, color = [0, 0, 0])        # grey background

        # dynamic components variables initialization
        self.k          = np.array([np.nan])                                    # block counter
        self.c          = np.array([np.nan])                                    # contrast difference
        self.o          = np.array([np.nan])                                    # observation
        self.b          = np.full((1,2), np.nan)                                # belief state    
        self.v          = np.full((1,2), np.nan)                                # decision value 
        self.d          = np.array([np.nan])                                    # decision
        self.r          = np.array([np.nan])                                    # reward
        self.rt         = np.array([np.nan])                                    # response time

    # agent functions
    # -------------------------------------------------------------------------
    def start_task(self):
        
        """
        Task start display until space bar pressed
        """
        
        info    = "Press space bar to start the experiment"
        text    = visual.TextStim(self.win, text = info, height = self.fs)
        text.draw() 
        self.win.flip()
        event.waitKeys(keyList=["space"])
    
    def start_block(self):
        
        """
        Block start display until space bar pressed
        """
        self.rr = 0
        info    = "Press space bar to start a new block"
        text    = visual.TextStim(self.win, text = info, height = self.fs)
        text.draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])

    def action(self):

        """
        Stimulus presentation and agent action collection.
        """
       
        # stimulus preparation
        stim    = f"gb_{int(self.c >= 0)}_{self.c:.2f}.png"                     # file name        
        file    = self.sdir / stim                                              # file path
        image   = visual.ImageStim(self.win, image= file)                       # load stimulus
        image.draw()                                                            # draw stimulus
        text    = visual.TextStim(self.win, text = "+", pos=(0, 0))                         # fixation cross
        text.draw()                                                             # draw fixation cross        

        # stimulus presentation and  action collection
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
        text_reward = visual.TextStim(self.win, text = f"+{self.r}")            # reward
        text_reward.draw()
        self.win.flip()
        core.wait(self.ltime)


    def end_task(self):

        """
        Experiment finalization
        """
        text = visual.TextStim(self.win, text="Task completed")
        text.draw()
        self.win.flip()
        core.wait(2.0)
        self.win.close()
