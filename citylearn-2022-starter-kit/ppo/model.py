import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from functools import partial

# Actor 
def init_weights(module, gain):
    """
    Orthogonal initialization
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class PPO_Actor(nn.Module):
    def __init__(self, obs_dim,act_dim):
        super(PPO_Actor,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,act_dim)
        )

        self.log_sigma = nn.parameter.Parameter(torch.zeros(act_dim))
        self.distribution = torch.distributions.Normal
        module_gains = {
            self.layers[0] : np.sqrt(2),
            self.layers[2] : np.sqrt(2),
            self.layers[4] : 0.01
        }

        for module, gain in module_gains.items():
            module.apply(partial(init_weights, gain=gain))


    def forward(self,x):     
        if isinstance(x,np.ndarray):
            x = torch.Tensor(x,dtype = torch.float)
        
        mu = self.layers(x)
        sigma = torch.exp(self.log_sigma)
        dist = self.distribution(mu,sigma)
        
        return dist

class PPO_Critic(nn.Module):
    def __init__(self, obs_dim):
        super(PPO_Critic,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

        module_gains = {
            self.layers[0] : np.sqrt(2),
            self.layers[2] : np.sqrt(2),
            self.layers[4] : 0.01
        }

        for module, gain in module_gains.items():
            module.apply(partial(init_weights, gain=gain))


    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = torch.Tensor(x,dtype = torch.float)
        
        values = self.layers(x)  
        return values.view(-1)