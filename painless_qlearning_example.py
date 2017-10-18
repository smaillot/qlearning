# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:30:09 2017

@author: Stéphane Maillot
"""

from QLearning import *


def apply_model(state, action):

    model = np.array([[0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 1, 0, 0], 
                      [0, 0, 1, 0, 1, 0], 
                      [1, 0, 0, 1, 0, 1], 
                      [0, 1, 0, 0, 1, 1]])
    
    if model[state, action]:
        
        return action
    
    else:
    
        return state

R0 = np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 100],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 100],
               [0, 0, 0, 0, 0, 100]])

learning = QLearning(range(0,6), range(0,6), R0, 0.5, 1)

state = 2

for i in range(0,1000):
    
    action = learning.choose_action(state, 1 - float(i)/100)
    next_state = apply_model(state, action)
    learning.update_model(state, action, next_state)
    if mod(i, 100) == 0:
        state = 2
    else:
        state = next_state
    
    #print(state)

print(np.round(learning.Q))