# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:16:13 2017

@author: St&phane Maillot
"""

import numpy as np
from QLearning import *
import GridWorld as gw
import matplotlib.pylab as plt

def mapper(x, dim):
    
    return [x/dim[1], np.mod(x, dim[1])]

def unmapper(x, dim):
    
    return x[0] * dim[1] + x[1]

model = np.array([[0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 1, 0, 0], 
                      [0, 0, 1, 0, 1, 0], 
                      [1, 0, 0, 1, 0, 1], 
                      [0, 1, 0, 0, 1, 1]])
    
dim = np.shape(model)

def apply_model(state, action):
    
    cord = mapper(state, dim)
    
    if (cord[0] == 0 and action == 1) or (cord[0] == dim[0] - 1 and action == 3) or (cord[1] == 0 and action == 2) or (cord[1] == dim[1] - 1 and action == 0):
        
        return state
    
    elif (action == 0 and model[cord[0], cord[1] + 1]) or (action == 1 and model[cord[0] - 1, cord[1]]) or (action == 2 and model[cord[0], cord[1] - 1]) or (action == 3 and model[cord[0] + 1, cord[1]]):
        
        return state
    
    else:   
        
        if action == 0:
            
            return unmapper([cord[0], cord[1] + 1], dim)
                           
        elif action == 1:
            
            return unmapper([cord[0] - 1, cord[1]], dim)
                           
        elif action == 2:
            
            return unmapper([cord[0], cord[1] - 1], dim)
                           
        else:
            
            return unmapper([cord[0] + 1, cord[1]], dim)
                          

R0 = np.zeros([36, 4])
R0[32, 0] = 100

learning = QLearning(range(0,36), range(0,4), R0, 0.5, 1)

state = 23

for i in range(0,10000):
    
    action = learning.choose_action(state, 1 - float(i)/100)
    next_state = apply_model(state, action)
    learning.update_model(state, action, next_state)
    if mod(i, 100) == 0:
        state = 23
    else:
        state = next_state
    
    plt.clf()
gw.DrawMap(np.reshape(np.max(learning.Q, 1), [6, 6]), model)
    