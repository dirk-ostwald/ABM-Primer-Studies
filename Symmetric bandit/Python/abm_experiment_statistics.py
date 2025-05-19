""" 
This scripts evaluates and visualizes the descriptive statistics of the 
perceptual choice and SARC learning experimental data.

Author - Dirk Ostwald
"""

# initialization
# -----------------------------------------------------------------------------
# general utility import
import numpy as np                                                              # NumPy
import matplotlib.pyplot as plt                                                 # MatPlotLib
import matplotlib.gridspec as gridspec                                          # subplots
plt.close("all")                                                                # close all figures
import sys                                                                      # system tools
import matplotlib.image as mpimg                                                # image reading
from pathlib import Path                                                        # path management

# directory management
wdir        = Path.cwd()                                                        # working directory
udir        = wdir / "abm"                                                      # project utilities directory
rdir        = wdir.parent                                                       # project root directory
ddir        = rdir / "Data" / "Experiment"                                      # data source directory
fdir        = rdir / "Figures"                                                  # figure directory

# ABM project utility import
sys.path.append(str(udir))                                                      # utilities directory addition to Python path
from abm_structure import abm_structure                                         # type: ignore # Python structure simulation utility
from abm_abm import abm_abm                                                     # type: ignore # ABM simulation utility
from abm_description import abm_description                                     # type: ignore # descriptive statistics utility
from abm_figures import abm_figures                                             # type: ignore # visualization utilities

 
# descriptive statistics
# -----------------------------------------------------------------------------     
idxs        = np.array([5,8,10])                                                # participant indices
n           = len(idxs)                                                         # number of data sets
sta         = abm_structure()                                                   # structure initialization
sta.ddir    = ddir                                                              # source data directory
sta.idxs    = idxs                                                              # participant indices
sta.K       = 10                                                                # number of blocks
sta.T       = 30                                                                # number of trials per block
sta         = abm_description(sta)                                              # run descriptive statistics           

# group economic choice data visualization
# ----------------------------------------------------------------------------
# figure settings
plt, blue, red      = abm_figures(plt)
fig                 = plt.figure(figsize = (12,10))                                   
gs                  = gridspec.GridSpec(2,2)                                         
ax                  = {}    

# paradigm
ax[0]               = plt.subplot(gs[0,:])   
ax[0].imshow(mpimg.imread(fdir / 'paradigm.png'))
ax[0].axis('off')  

# group accuracy
x                   = np.arange(1,n+1)                                                 # participant indices        
ax[1]               = plt.subplot(gs[1,0])                                                  
ax[1].plot(         x,
                    np.full(n, sta.r_avg_mean), 
                    ls = '-', 
                    color = red[1], 
                    label = 'Group mean reward')
ax[1].plot(         x,
                    np.full(n, sta.a_avg_mean), 
                    ls = '-', 
                    color = 'gray', 
                    label = 'Group mean maximizing action')
ax[1].plot(         x,
                    sta.r_avg,
                    marker = 'o', 
                    ls = '', 
                    color = red[2], 
                    clip_on = False, 
                    label = 'Participant mean reward')
ax[1].plot(         x,
                    sta.a_avg,
                    marker = 'o', 
                    ls = '', 
                    color = 'gray', 
                    clip_on = False, 
                    label = 'Participant maximizing action')
ax[1].fill_between( x,
                    np.full(n, sta.r_avg_mean) - np.full(n, sta.r_avg_sd),
                    np.full(n, sta.r_avg_mean) + np.full(n, sta.r_avg_sd),
                    facecolor    = red[3],
                    edgecolor    = red[3],
                    alpha        = 0.2,
                    label        = 'Standard deviation reward')
ax[1].fill_between( x,
                    np.full(n, sta.a_avg_mean) - np.full(n, sta.a_avg_sd),
                    np.full(n, sta.a_avg_mean) + np.full(n, sta.a_avg_sd),
                    facecolor    = 'lightgray',
                    edgecolor    = 'lightgray',
                    alpha        = 0.2,
                    label        = 'Standard deviation maximizing action')
ax[1].set_ylim(.5, 1.)
ax[1].set_xlim(0,51)
ax[1].set_title('Average rewards and maximzing actions', fontsize = 22)
ax[1].set_xlabel('Participant', fontsize = 18)
ax[1].tick_params(labelsize = 14)
ax[1].grid(True, linewidth = .5, color = [.9,.9,.9])
ax[1].legend(loc = 'lower right', fontsize = 8, frameon = True)

# group learning curves
print(sta.a_lrn_mean)
ax[2] = plt.subplot(gs[1,1])                                                  
ax[2].plot(         sta.r_lrn_x, 
                    sta.r_lrn_mean, 
                    ls = '-', 
                    color = red[1], 
                    label = 'Group mean reward')
ax[2].plot(         sta.a_lrn_x, 
                    sta.a_lrn_mean, 
                    ls = '-', 
                    color = 'gray', 
                    label = 'Group mean maximizing action')
ax[2].fill_between( sta.r_lrn_x, 
                    sta.r_lrn_mean - sta.r_lrn_sem,
                    sta.r_lrn_mean + sta.r_lrn_sem,
                    facecolor    = red[3],
                    edgecolor    = red[3],
                    alpha        = 0.2,
                    label        = 'SEM reward')
ax[2].fill_between( sta.a_lrn_x, 
                    sta.a_lrn_mean - sta.a_lrn_sem,
                    sta.a_lrn_mean + sta.a_lrn_sem,
                    facecolor    = 'lightgray',
                    edgecolor    = 'lightgray',
                    alpha        = 0.2,
                    label        = 'SEM maximizing action')
ax[2].set_ylim(.5, 1.)
ax[2].set_xlim(1,25)
ax[2].set_title('Reward and maxizing action Learning curves', fontsize = 20)
ax[2].set_xlabel('Trial', fontsize = 18)
ax[2].tick_params(labelsize = 14)
ax[2].grid(True, linewidth = .5, color = [.9,.9,.9])
ax[2].legend(loc = 'lower right', fontsize = 8, frameon = True)

# subplot labels and figure saving
labels = ['A', 'B', 'C']
for i, label in enumerate(labels):
    ax[i].text(
    -0.1, 
     1.10, 
    label,
    transform   = ax[i].transAxes,
    fontsize    = 22,
    fontweight  = 'bold',
    va          = 'top', 
    ha          = 'left')
fig.tight_layout()
fig.savefig(fdir / "abm_sb_group.pdf", dpi = 300, format = "pdf")

