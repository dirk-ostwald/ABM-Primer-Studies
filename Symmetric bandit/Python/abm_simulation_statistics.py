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
from pathlib import Path                                                        # path management


# directory management
wdir        = Path.cwd()                                                        # working directory
udir        = wdir / "abm"                                                      # project utilities directory
rdir        = wdir.parent                                                       # project root directory
sdir        = rdir / "Data" / "Simulations"                                     # data source directory
fdir        = rdir / "Figures"                                                  # figure directory

# ABM project utility import
sys.path.append(str(udir))                                                      # utilities directory addition to Python path
from abm_structure import abm_structure                                         # type: ignore # Python structure simulation utility
from abm_abm import abm_abm                                                     # type: ignore # ABM simulation utility
from abm_description import abm_description                                     # type: ignore # descriptive statistics utility
from abm_figures import abm_figures                                             # type: ignore # visualization utilities

# toy data generation
# -----------------------------------------------------------------------------
n           = 50                                                               # number of data sets
I           = np.arange(n)                                                      # participant indices
S           = rdir / "Data" / "Simulations"                                     # source data directory
P           = abm_structure()                                                   # structure initialization
P.mode      = "gen"                                                             # generative mode
P.agent     = "A"                                                               # agent name                          
P.tau       = 0.05                                                              # action noise
P.K         = 10                                                                # number of blocks
P.T         = 30                                                                # number of trials per block
P.dir       = sdir                                                              # data directory
for i in I:                                                                     # data set iterations
    P.P         = i                                                             # participant index 
    P.file      = P.dir / f"{"sub-" + str(i).zfill(3)}.csv"                     # filename    
    P           = abm_abm(P)                                                    # run simulation


# descriptive statistics
# -----------------------------------------------------------------------------     
sta         = abm_structure()                                                   # structure initialization
sta.S       = S                                                                 # source data directory
sta.I       = I                                                                 # participant indices
sta.K       = P.K                                                               # number of blocks
sta.T       = P.T                                                                # number of trials per block                                             
sta         = abm_description(sta)                                              # run descriptive statistics        

# group economic choice data visualization
# ----------------------------------------------------------------------------
# figure settings
plt, blue, red      = abm_figures(plt)
fig                 = plt.figure(figsize = (12,5))                                   
gs                  = gridspec.GridSpec(1,2)                                         
ax                  = {}    

# group accuracy
ax[0]               = plt.subplot(gs[0,0])                                                  
ax[0].plot(         I, 
                    np.full(n, sta.r_avg_mean), 
                    ls = '-', 
                    color = red[1], 
                    label = 'Group mean reward')
ax[0].plot(         I, 
                    np.full(n, sta.a_avg_mean), 
                    ls = '-', 
                    color = 'gray', 
                    label = 'Group mean maximizing action')
ax[0].plot(         I, 
                    sta.r_avg,
                    marker = 'o', 
                    ls = '', 
                    color = red[2], 
                    clip_on = False, 
                    label = 'Participant mean reward')
ax[0].plot(         I, 
                    sta.a_avg,
                    marker = 'o', 
                    ls = '', 
                    color = 'gray', 
                    clip_on = False, 
                    label = 'Participant maximizing action')
ax[0].fill_between( I, 
                    np.full(n, sta.r_avg_mean) - np.full(n, sta.r_avg_sd),
                    np.full(n, sta.r_avg_mean) + np.full(n, sta.r_avg_sd),
                    facecolor    = red[3],
                    edgecolor    = red[3],
                    alpha        = 0.2,
                    label        = 'Standard deviation reward')
ax[0].fill_between( I, 
                    np.full(n, sta.a_avg_mean) - np.full(n, sta.a_avg_sd),
                    np.full(n, sta.a_avg_mean) + np.full(n, sta.a_avg_sd),
                    facecolor    = 'lightgray',
                    edgecolor    = 'lightgray',
                    alpha        = 0.2,
                    label        = 'Standard deviation maximizing action')
ax[0].set_ylim(.5, 1.)
ax[0].set_xlim(0,51)
ax[0].set_title('Average rewards and maximzing actions', fontsize = 22)
ax[0].set_xlabel('Participant', fontsize = 18)
ax[0].tick_params(labelsize = 14)
ax[0].grid(True, linewidth = .5, color = [.9,.9,.9])
ax[0].legend(loc = 'lower right', fontsize = 8, frameon = True)

# group learning curves
print(sta.a_lrn_mean)
ax[1] = plt.subplot(gs[0,1])                                                  
ax[1].plot(         sta.r_lrn_x, 
                    sta.r_lrn_mean, 
                    ls = '-', 
                    color = red[1], 
                    label = 'Group mean reward')
ax[1].plot(         sta.a_lrn_x, 
                    sta.a_lrn_mean, 
                    ls = '-', 
                    color = 'gray', 
                    label = 'Group mean maximizing action')
ax[1].fill_between( sta.r_lrn_x, 
                    sta.r_lrn_mean - sta.r_lrn_sem,
                    sta.r_lrn_mean + sta.r_lrn_sem,
                    facecolor    = red[3],
                    edgecolor    = red[3],
                    alpha        = 0.2,
                    label        = 'SEM reward')
ax[1].fill_between( sta.a_lrn_x, 
                    sta.a_lrn_mean - sta.a_lrn_sem,
                    sta.a_lrn_mean + sta.a_lrn_sem,
                    facecolor    = 'lightgray',
                    edgecolor    = 'lightgray',
                    alpha        = 0.2,
                    label        = 'SEM maximizing action')
ax[0].set_ylim(.5, 1.)
ax[1].set_xlim(1,25)
ax[1].set_title('Reward and maxizing action Learning curves', fontsize = 20)
ax[1].set_xlabel('Trial', fontsize = 18)
ax[1].tick_params(labelsize = 14)
ax[1].grid(True, linewidth = .5, color = [.9,.9,.9])
ax[1].legend(loc = 'lower right', fontsize = 8, frameon = True)

fig.tight_layout()
fig.savefig(fdir / "abm_sb_group.pdf", dpi = 300, format = "pdf")

# # participant-level economic choice data visualization
# # -----------------------------------------------------------------------------
# # participant-level economic action accuracy learning curves
# fig             = plt.figure(figsize = (12,12))                                 # figure initialization and figure size
# nrows           = 8                                                             # number of rows
# ncols           = 7                                                             # number of columns
# gs              = gridspec.GridSpec(8,7)                                        # subplot layout
# ax              = {}                                                            # axes dictionary initialization          
# idx             = 0                                                             # linear index initialization

# for i in range(nrows):
#     for j in range(ncols):
#         ax[idx] = plt.subplot(gs[i,j])
#         ax[idx].plot(sta.ae_lrn_x, sta.ae_lrn[idx,:], ls = '-', color = red[2])
#         ax[idx].set_title('P{0:1}, {1:1.2f}'.format(idx,sta.ae_acc[idx]), fontsize = 8)
#         ax[idx].set_ylim(.45, 1.05)
#         ax[idx].set_xlim(1,25)
#         ax[idx].tick_params(labelsize = 8)
#         ax[idx].grid(True, linewidth = .5, color = [.9,.9,.9])
#         idx = idx + 1
#         if idx >= n:
#             break
#     if idx >= n:
#         break

# fig.tight_layout()
# fig.savefig(os.path.join(fdir, 'abm_sl_participants.pdf'), dpi = 300, format = 'pdf')


