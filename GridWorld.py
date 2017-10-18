# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 18:07:56 2017

@author: Stéphane Maillot
"""

import matplotlib.pylab as plt
import numpy as np


def DrawMap(Q, model):
    
    M = Q
    M -= np.min([np.min(Q), 0])
    if np.max(M) != 0:
        M /= np.max(M)
    M[np.where(model)] = -1
    plt.pcolor(Q, cmap="brg")
    plt.colorbar()