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
        self.network_critic = nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,s,a):
        q_value = self.network_critic(torch.cat([s, a], 1))
        return q_value

class TD3_MLP_ACTOR(nn.Module):
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

class TD3_MLP_CRITIC(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super().__init__()
        #critic
        self.network_critic_1 = nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

        self.network_critic_2 = nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,s,a):
        q_value1 = self.network_critic_1(torch.cat([s, a], 1))
        q_value2 = self.network_critic_2(torch.cat([s, a], 1))
        return q_value1

