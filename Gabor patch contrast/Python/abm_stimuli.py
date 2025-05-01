""" 
This scripts creates the Gabor patch contrast discrimination task stimulus set. 
Author - Dirk Ostwald
"""

# initialization
# -----------------------------------------------------------------------------
import numpy as np                                                              # numpy
import matplotlib.pyplot as plt                                                 # matplotlib
from pathlib import Path                                                        # path management
import os                                                                       # operating system module

# Gabor function
# -----------------------------------------------------------------------------
def gabor(x, y, sigma, lambd, theta, phi, gamma):
    
    """
    This funcion evaluates the Gabor function at (x, y).
    """
    xp  = x * np.cos(phi) + y * np.sin(phi)
    yp  = -x * np.sin(phi) + y * np.cos(phi)
    gxy = gamma * np.exp(-(xp**2 + yp**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xp / lambd + theta)
    return gxy

# stimulus creation
# -----------------------------------------------------------------------------
# directory management
wdir        = Path.cwd()                                                        # working directory
udir        = wdir / "abm"                                                      # project utilities directory
rdir        = wdir.parent                                                       # project root directory
sdir        = rdir / "Stimuli"                                                  # stimulus directory
os.makedirs(sdir, exist_ok = True)

# Constants
r       = 1000                                                                  # image resolution
x       = np.linspace(-50, 50, r)                                               # x-support
y       = np.linspace(-50, 50, r)                                               # y-support
X, Y    = np.meshgrid(x, y)                                                     # (x,y)-support
phi     = 0.00                                                                  # Gabor patch orientation
lambd   = 15.0                                                                  # sinusisoidal factor wavelength
theta   = 0.00                                                                  # sinuisoidal phase offset
sigma   = 15.0                                                                  # Gaussian envelope size
mu      = 0.50                                                                  # mean stimulus amplitude scaling
nkappa  = 1.00                                                                  # normalized contrast difference parameter
kappa   = 0.50                                                                  # maximal contrast difference parameter
dc      = 0.01                                                                  # normalized contrast difference step size
cs      = np.linspace(-nkappa, nkappa, int((nkappa - (-nkappa)) / dc) + 1)      # normalized contrast difference set 

# contrast difference iterations
for c in cs:

    # amplitude scalings
    gamma_l     = mu - 0.5*(c*kappa)                                            # left Gabor patch
    gamma_r     = gamma_l + (c*kappa)                                           # right Gabor patch    
    gamma       = np.array([gamma_l, gamma_r])                                  # amplitude scaling array
    s           = int(c*kappa >= 0)                                             # state

    # figure initialization
    fig, axs = plt.subplots(1, 2, figsize = (12, 4))
    fig.patch.set_facecolor((0.5, 0.5, 0.5))

    # Gabor patch iterations
    for g in range(2):
        gp  = gabor(X, Y, sigma, lambd, theta, phi, gamma[g])
        ax  = axs[g]
        im  = ax.imshow(gp, cmap='gray', extent=[-50, 50, -50, 50], vmin=-1, vmax=1)
        ax.axis('off')

    # figure saving       

    slab = f"gb_{s}_{c:.2f}.png"
    plt.savefig(os.path.join(sdir, slab), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
