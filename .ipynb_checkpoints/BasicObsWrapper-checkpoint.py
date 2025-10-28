#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:56:19 2021
@author: oliver
"""
import gymnasium as gym
from gymnasium import spaces

class SimpleObsWrapper(gym.ObservationWrapper):
    """Simple gridworld returning a single value encoding."""

    def __init__(self, env, max_env_steps=50):
        super().__init__(env)
        
        # Return a single number encoding cell and direction of the agent
        self.observation_space = spaces.Box(
            low=0,
            high=(self.env.unwrapped.width-2) * (self.env.unwrapped.height-2) * 4, # 4 directions
            shape=(1,),  # number of cells
            dtype='uint8'
        )
        self.unwrapped.max_steps = max_env_steps

    def observation(self, obs):
        # this method is called in the step() function to get the observation
        # we provide code that gets the grid state and places the agent in it
        env = self.unwrapped
        state = (env.agent_pos[0] - 1) + (
            (self.env.unwrapped.width - 2) * (env.agent_pos[1] - 1) +
            env.agent_dir * (self.env.unwrapped.width - 2) * (self.env.unwrapped.height - 2)
        )
        return state