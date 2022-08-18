# Pytorch models code for DDPG,TD3 and SAC
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPG_MLP_ACTOR(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super().__init__()
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Tanh()
        )
    
    def forward(self,s):
        action = self.network_actor(s)
        return action

class DDPG_MLP_CRITIC(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super().__init__()
        #critic
        self.network_actor = nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,s,a):
        q_value = self.network_actor(torch.cat([s, a], 1))
        return q_value