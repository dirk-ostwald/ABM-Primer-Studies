""" 
This scripts evaluates and visualizes the descriptive statistics of the Gabor
patch contrast discrimination experimental data.

Author - Dirk Ostwald
"""

# initialization
# -----------------------------------------------------------------------------
# general utility import
import numpy as np                                                              # NumPy
import matplotlib.pyplot as plt                                                 # MatPlotLib
import matplotlib.gridspec as gridspec                                          # subplots
import matplotlib.image as mpimg                                                # image reading
plt.close("all")                                                                # close all figures
import sys                                                                      # system tools
from pathlib import Path                                                        # path management

# directory management
wdir        = Path.cwd()                                                        # working directory
udir        = wdir / "abm"                                                      # project utilities directory
rdir        = wdir.parent                                                       # project root directory
ddir        = rdir / "Data" / "Simulations"                                     # data source directory
fdir        = rdir / "Figures"                                                  # figure directory

# ABM project utility import
sys.path.append(str(udir))                                                      # utilities directory addition to Python path
from abm_structure import abm_structure                                         # Python structure simulation utility
from abm_abm import abm_abm                                                     # ABM simulation utility
from abm_description import abm_description                                     # descriptive statistics utility
from abm_figures import abm_figures                                             # visualization utilities


# descriptive statistics
# -----------------------------------------------------------------------------     
n           = 50                                                                 # number of data sets
idxs        = range(n)                                                          # participant indices
sta         = abm_structure()                                                   # structure initialization
sta.ddir    = ddir                                                              # source data directory
sta.idxs    = idxs                                                              # participant indices
sta         = abm_description(sta)                                              # run descriptive statistics        

# group perceptual choice data visualization
# -----------------------------------------------------------------------------
plt, blue, red      = abm_figures(plt)
fig                 = plt.figure(figsize = (18,5))                                   
gs                  = gridspec.GridSpec(1,3)                                         
ax                  = {}                     

# paradigm
ax[0]               = plt.subplot(gs[0,0])   
ax[0].imshow(mpimg.imread(fdir / 'paradigm.png'))
ax[0].axis('off')  

# group accuracy
ax[1]               = plt.subplot(gs[0,1])                                                  
ax[1].plot(         idxs, 
                    np.full(n, sta.a_acc_mean), 
                    ls      = '-', 
                    color   = blue[1] , 
                    label   = 'Group mean')
ax[1].plot(         idxs, 
                    sta.a_acc, 
                    marker  = 'o', 
                    ls      = '', 
                    color   = blue[2], 
                    clip_on = False, 
                    label   = 'Participant')
ax[1].fill_between( idxs, 
                    np.full(n, sta.a_acc_mean) - np.full(n, sta.a_acc_sd),
                    np.full(n, sta.a_acc_mean) + np.full(n, sta.a_acc_sd),
                    facecolor = blue[3],
                    edgecolor = blue[3],
                    alpha   = 0.2,
                    label   = 'Standard deviation')
ax[1].set_ylim(.5, 1.)
ax[1].set_xlim(0,n+1)
ax[1].set_title('Action accuracy', fontsize = 22)
ax[1].set_xlabel('Participant', fontsize = 18)
ax[1].tick_params(labelsize = 14)
ax[1].legend(loc = 'lower right', fontsize = 12, frameon = True)
ax[1].grid(True, linewidth = .5, color = [.9,.9,.9])

# group psychometric function
ax[2]               = plt.subplot(gs[0,2])                                                  
ax[2].plot(         sta.a_psy_x, 
                    sta.a_psy_mean, 
                    ls           = '-', 
                    color        =  blue[1], 
                    label        = 'Group mean')
ax[2].fill_between( sta.a_psy_x, 
                    sta.a_psy_mean - sta.a_psy_sd,
                    sta.a_psy_mean + sta.a_psy_sd,
                    facecolor    = blue[3],
                    edgecolor    = blue[3],
                    alpha        = 0.2,
                    label        = 'Standard deviation')
ax[2].set_ylim(-0.05, 1.05)
ax[2].set_xlim(-1.05, 1.05)
ax[2].set_title('Psychometric function', fontsize = 22)
ax[2].set_xlabel('Normalized contrast difference', fontsize = 18)
ax[2].tick_params(labelsize = 14)
ax[2].grid(True, linewidth = .5, color = [.9,.9,.9])
ax[2].legend(loc = 'lower right', fontsize = 12, frameon = True)

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
fig.savefig(fdir / "abm_gpc_group.pdf", dpi = 400, format = "pdf")

# participant-level perceptual choice data visualization
# -----------------------------------------------------------------------------
fig             = plt.figure(figsize = (12,12))                                  
nrows           = 8                                                              
ncols           = 7                                                              
gs              = gridspec.GridSpec(8,7)                                         
ax              = {}                                                                      
idx             = 0                                                              
for i in range(nrows):
    for j in range(ncols):
        ax[idx] = plt.subplot(gs[i,j])
        ax[idx].plot(sta.a_psy_x, sta.a_psy[idx,:], ls = '-', color = blue[2])
        ax[idx].set_title('P{0:1}, Accuracy {1:1.2f}'.format(idx,sta.a_acc[idx]), fontsize = 8)
        ax[idx].set_ylim(-0.05, 1.05)
        ax[idx].set_xlim(-1.05, 1.05)
        ax[idx].tick_params(labelsize = 8)
        ax[idx].grid(True, linewidth = .5, color = [.9,.9,.9])
        idx = idx + 1
        if idx >= n:
            break
    if idx >= n:
        break
fig.tight_layout()
fig.savefig(fdir / "abm_gpc_participants.pdf", dpi = 400, format = "pdf")

# # group economic choice data visualization
# # ----------------------------------------------------------------------------
# # figure settings
# plt, blue, red      = abm_figures(plt)
# fig                 = plt.figure(figsize = (12,5))                                   
# gs                  = gridspec.GridSpec(1,2)                                         
# ax                  = {}    

# # group accuracy
# ax[0]               = plt.subplot(gs[0,0])                                                  
# ax[0].plot(         P, 
#                     sta.ae_acc_mean*np.ones([n]), 
#                     ls = '-', 
#                     color = red[1], 
#                     label = 'Group mean')
# ax[0].plot(         P, 
#                     sta.ae_acc,
#                     marker = 'o', 
#                     ls = '', 
#                     color = red[2], 
#                     clip_on = False, 
#                     label = 'Participant')
# ax[0].fill_between( P, 
#                     sta.ae_acc_mean*np.ones([n]) - sta.ae_acc_sd*np.ones([n]),
#                     sta.ae_acc_mean*np.ones([n]) + sta.ae_acc_sd*np.ones([n]),
#                     facecolor    = red[3],
#                     edgecolor    = red[3],
#                     alpha        = 0.2,
#                     label        = 'Standard deviation')
# ax[0].set_ylim(.5, 1.)
# ax[0].set_xlim(0,51)
# ax[0].set_title('Action accuracy', fontsize = 22)
# ax[0].set_xlabel('Participant', fontsize = 18)
# ax[0].tick_params(labelsize = 14)
# ax[0].grid(True, linewidth = .5, color = [.9,.9,.9])
# ax[0].legend(loc = 'lower right', fontsize = 12, frameon = True)

# # group learning curve
# ax[1] = plt.subplot(gs[0,1])                                                  
# ax[1].plot(         sta.ae_lrn_x, 
#                     sta.ae_lrn_mean, 
#                     ls = '-', 
#                     color = red[1], 
#                     label = 'Group mean')
# ax[1].fill_between( sta.ae_lrn_x, 
#                     sta.ae_lrn_mean - sta.ae_lrn_sd,
#                     sta.ae_lrn_mean + sta.ae_lrn_sd,
#                     facecolor    = red[3],
#                     edgecolor    = red[3],
#                     alpha        = 0.2,
#                     label        = 'Standard deviation')
# ax[0].set_ylim(.5, 1.)
# ax[1].set_xlim(1,25)
# ax[1].set_title('Learning curve', fontsize = 20)
# ax[1].set_xlabel('Trial', fontsize = 18)
# ax[1].tick_params(labelsize = 14)
# ax[1].grid(True, linewidth = .5, color = [.9,.9,.9])
# ax[1].legend(loc = 'lower right', fontsize = 12, frameon = True)

# fig.tight_layout()
# fig.savefig(os.path.join(fdir, 'abm_sl_group.pdf'), dpi = 300, format = 'pdf')

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


