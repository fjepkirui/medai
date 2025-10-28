#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:36:41 2020

@author: oliver
"""

import random
import gym
import numpy as np
import matplotlib.pyplot as plt

def observation_to_state(env, obs, states = (40,40)):
    """
    Convert observation from environment into a discrete state.
    
    Parameters
    ----------
    env : gym env
        Reference to the current environment.
    obs : tuple or array 
        An observation for the agent.
    states : tuple, optional
        For each dimension, the number of bins used. The default is (40,40).

    Returns
    -------
    A tuple containing the state corresponding to the observation.

    """    
    ##
    ## your code here
    ##
    
    return s0, s1


### RL below this line

def choose(env, q, state, epsilon):
    """
    epsilon-greedy action selection. Greedy action selection if epsilon == 0.

    Parameters
    ----------
    env : gym env
        Reference to the current environment
    q : np.array (state_1 x .. x state_n x actions)
        A (n+1) dimensional q table:
            (n state dimensions + 1 action dimension)
            probability of taking an action
    state : tuple
        2-dimensional integer tuple describing the current state of the agent.
    epsilon : float
        Probability of taking a random action.
        If epsilon == 0, we use (max) greedy action selection.
        If epsilon > 0, we choose epsilon-greedy and according to probability

    Returns
    -------
    int
        The chosen action.

    """
    assert len(q.shape)-1 == len(state), "number of state dimensions in "
    "q-table should match dimensions in state parameter"

    s0, s1 = state
    
    if epsilon == 0:
        # return best action (deterministically)
        return np.argmax(q[s0, s1])
    
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(env.action_space.n)

    logits = q[s0, s1]    
    logits_exp = np.exp(logits)
    probabilities = logits_exp / sum(logits_exp)
    
    return np.random.choice(env.action_space.n, p = probabilities)


def update_model(model, actions, state, action, reward, nextstate):
    model[(state, action)] = (reward, nextstate)
    
    if state in actions:
        if action not in actions[state]:
            actions[state].append(action)
    else:
        actions[state] = [action]


def select_from_model(model, actions):
    ##
    ## your code here
    ##
    return state, action, model[(state, action)]
    


def qlearn(env, alpha, epsilon, gamma, max_steps,
           num_episodes = 2000, num_states = (40,40)):
    """
    Tabular Q-Learning for gym

    Parameters
    ----------
    env : gym env
        Reference to the current environment
    alpha : float 
        an initial learning rate; reduced for every episode down to a minimum.
    epsilon : float
        a probability for choosing a random action (0 <= epsilon <= 1).
    terminal : list
        a list of terminal states
    gamma : float
        a discount value (0 < gamma < 1).
    num_episodes : int, optional
        Number of episodes to train. The default is 5000.
    num_states : tuple, optional
        Number of states in each observation dimension. The default is (40,40).

    Returns
    -------
    q : array (num_states x num_actions)
        state-action value table.

    """
    q = np.zeros((num_states[0], num_states[1], env.action_space.n))

    # model is an empty dictionary. 
    # It will map from (state, action) -> (reward, nextstate)
    model = dict()
    # actions is an empty dictionary
    # It will map from state -> set of actions
    actions = dict()
    # number of dyna steps
    dynasteps = 5

    avg_reward = 0
    avg_freq = 50
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        eta = max(0.005, alpha * (0.85 ** (i//100)))

        for t in range(max_steps):
            s0, s1 = observation_to_state(env, obs, num_states)
            action = choose(env, q, (s0, s1), epsilon)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            avg_reward += reward
            
            s2, s3 = observation_to_state(env, obs, num_states)
            
            q[s0, s1, action] = q[s0, s1, action] + eta * (reward + gamma *  np.max(q[s2,s3]) - q[s0, s1, action])
            
            ## dyna-q part:
            # your code here: update model 

            # execute planning updates
            for j in range(dynasteps):
                # your code: use model to select action 
                # your code: update q-table
           
            if done or truncated:
                break

            s0, s1 = s2, s3

        if i % avg_freq == 0:
            avg_reward /= avg_freq
            print('Iteration: %d (Dyna: %d steps) | Average total: %d | Last total: %d.' %(i+1, dynasteps, avg_reward, total_reward))
            avg_reward = 0

    return q, model, actions


def run_policy(env, q, max_steps, num_states = (40,40), render = False):

    obs, _ = env.reset()
    total_reward = 0
    for t in range(max_steps):
        state = observation_to_state(env, obs, num_states)
        action = choose(env, q, state, 0)
        obs, reward, done, truncated, _ = env.step(action)
        if render:
            plt.imshow(env.render())
            plt.show()
            
        total_reward += reward
        if done or truncated:
            break
    
    return total_reward


alpha = 1.0
epsilon = 0.05
gamma = 1.0
max_steps = 200
environment = gym.make('MountainCar-v0')

np.random.seed(0)
num_states = (40,40)
num_episodes = 5000

q, model, actions = qlearn(environment, alpha, epsilon, gamma, max_steps, 
                           num_episodes = num_episodes, num_states = num_states)
scores = [run_policy(environment, q, max_steps, num_states = num_states) for _ in range(100)]
print('Mean score: ', np.mean(scores))

run_policy(environment, q, max_steps, num_states = num_states, render = True)
