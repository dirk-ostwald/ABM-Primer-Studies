
""" 
This scripts runs the symmetric bandit task experiment for a single participant. 

Author - Dirk Ostwald
"""

# initialization
# -----------------------------------------------------------------------------
# general utility import
import sys                                                                      # system tools
from pathlib import Path                                                        # path management

# directory management
wdir        = Path.cwd()                                                        # working directory
udir        = wdir / "abm"                                                      # project utilities directory
rdir        = wdir.parent                                                       # project root directory
ddir        = rdir / "Data" / "Experiment"                                      # data directory

# ABM project utility import
sys.path.append(str(udir))                                                      # utilities directory addition to Python path
from abm_structure import abm_structure                                         # Python structure simulation utility
from abm_abm import abm_abm                                                     # ABM simulation utility


# parameter structure specification
# -----------------------------------------------------------------------------
P           = abm_structure()                                                   # structure initialization
P.mode      = "gen"                                                             # mode of operation       
P.agent     = "H"                                                               # agent name
P.P         = 1                                                                 # participant ID 
P.K         = 1                                                                 # number of blocks
P.T         = 10                                                                # number of trials per block
P.ddir      = ddir                                                              # data directory
P.pid       = "sub-" + str(P.P).zfill(3)                                        # participant ID
P.file      = P.ddir / f"{P.pid}.csv"                                           # filename    

# run experiment
# -----------------------------------------------------------------------------
P           = abm_abm(P)                                                        # run experiment