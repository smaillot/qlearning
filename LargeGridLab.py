# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:16:13 2017

@author: Stéphane Maillot
"""

import numpy as np
from QLearning import *
import GridWorld as gw
import matplotlib.pylab as plt

def mapper(x, dim):
    
    return [x/dim[1], np.mod(x, dim[1])]

def unmapper(x, dim):
    
    return x[0] * dim[1] + x[1]

dim = [20, 60]
model = np.zeros(dim)
model[np.where(np.random.rand(dim[0], dim[1]) > 0.8)] = 1 

def find_free(model):
    
    free = np.where(model == 0)
    rand_pos = np.random.randint(len(free[0]))

    return unmapper([free[0][rand_pos], free[1][rand_pos]], dim)

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
                          

R0 = np.zeros([dim[0] * dim[1], 4])
goal = find_free(model)
R0[goal, 0] = 100

learning = QLearning(range(0,dim[0] * dim[1]), range(0,4), R0, 0.5, 1)

s0 = goal

while s0 == goal:
    
    s0 = find_free(model)

plt.close()
plt.figure(figsize=(dim[1]/2,dim[0]),facecolor='w') 

for _ in range(100):
    
    preview = np.zeros(dim)
            
    state = s0
    
    for _ in range(5000):
        
        action = learning.choose_action(state,0.1)
        next_state = apply_model(state, action)
        learning.update_model(state, action, next_state)        
        state = next_state
        
        
        
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.title('Q matrix coefficients')
    gw.DrawMap(np.reshape(np.max(learning.Q, 1), dim), model)
    plt.subplot(2, 1, 2)
    plt.title('Simulated learned path')
    s = s0
    for _ in range(dim[0] * dim[1]):
        
        s = apply_model(s, learning.choose_best_action(s))
        preview[tuple(mapper(s, dim))] = 0.5
        
        if s == goal:
            break
        
    preview[tuple(mapper(s0, dim))] = 1
    preview[tuple(mapper(goal, dim))] = 1
    gw.DrawMap(preview, model)
    plt.pause(0.001)
    

    
        