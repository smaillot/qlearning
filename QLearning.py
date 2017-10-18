# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:03:52 2017

@author: Stéphane Maillot
"""

import numpy as np


def nd2list(mat):
    
    if type(mat) == np.ndarray:
        
        return mat.tolist()
    
    elif type(mat) == list:
            
        return mat
    
def normalize(vec):
    
    s = np.sum(vec)
    v = vec
    if s != 0:
        v = map(float, vec) / s

    return v
               
    
def pick_random(probs):
    
    x = np.random.rand()
    s = np.sum(probs)
    if s != 0:
        p = normalize(probs)
    else:
        p = np.ones(len(probs)) / len(probs)
               
    p = np.cumsum(p)
            
    return np.min(np.where(p>x))

class QLearning:
    
    R = None
    Q = None
    states = None
    actions = None
    learning_rate = None
    discount_factor = None
    
    def __init__(self, states, actions, rewards, lr=1, df=0.5):
        
        self.states = list(set(states))
        self.actions = list(set(actions))
            
        self.Q = np.zeros(np.shape(rewards))
        self.R = np.array(rewards)
        self.learning_rate = lr
        self.discount_factor = df
        
    def get_actions(self, state):
    
        return self.Q[state,:]
    
    def choose_action(self, state, randomize=0):
    
        q_actions = self.get_actions(state)
        weighted = (1 - randomize) * normalize(q_actions) + randomize * normalize(np.ones(len(self.actions)))
        
        return self.actions[pick_random(weighted)]
    
    def choose_best_action(self, state):
        
        q_actions = self.get_actions(state)
        
        return np.where(q_actions == np.max(q_actions))[0][0]
        
    def update_model(self, state, action, next_state):
        
        alpha = self.learning_rate
        gamma = self.discount_factor
        self.Q[state, action] += alpha * ( self.R[state, action] + gamma * max( self.Q[next_state, :] ) - self.Q[state, action] )
        
        