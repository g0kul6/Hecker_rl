# This code will contain all the replay buffer and memory needed to sample trjectory(on-policy/off-policy)
import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition_DDPG = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))
class DDPG_Memory(object):
    def __init__(self,capacity):
        """
        Replay buffer for DDPG
        INPUT  : state,next_state,action,reward and done
        OUTPUT : batch of off-polocy experience
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        self.memory.append(Transition_DDPG(state, next_state, action, reward, done))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition_DDPG(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
    
Transition_TD3 = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))
class TD3_Memory(object):
    def __init__(self,capacity):
        """
        Replay buffer for TD3
        INPUT  : state,next_state,action,reward and done
        OUTPUT : batch of off-polocy experience
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        self.memory.append(Transition_TD3(state, next_state, action, reward, done))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition_TD3(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

class SAC_Memory(object):
    def __init__(self):
        """
        Memory for SAC
        INPUT  : 
        OUTPUT : 
        """
