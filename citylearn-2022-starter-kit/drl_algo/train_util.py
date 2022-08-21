# The training code for DDPG,TD3 and SAC
import os
import sys
path_dir = os.path.abspath(os.getcwd())
path_checkpoint = path_dir + "/checkpoint/"

from drl_algo.models import DDPG_MLP_ACTOR,DDPG_MLP_CRITIC,TD3_MLP_ACTOR,TD3_MLP_CRITIC
from drl_algo.memory import DDPG_Memory,TD3_Memory
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_ddpg_mlp(env,state_dim,action_dim,actor_lr,critic_lr,gamma,tau,episodes,random_steps,update_freq,batch_size,device):
    wandb.init(project="CITYCLEAN_RL",name="ddpg_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}".format(actor_lr,critic_lr,gamma,tau))
    #actor and actor target
    actor = DDPG_MLP_ACTOR(state_dim,action_dim,hidden_dim=120).to(device=device)
    actor_target = DDPG_MLP_ACTOR(state_dim,action_dim,hidden_dim=120).to(device=device)
    actor_target.load_state_dict(actor.state_dict())
    #critic and critic target
    critic = DDPG_MLP_CRITIC(state_dim,action_dim,hidden_dim=120).to(device=device)
    critic_target = DDPG_MLP_CRITIC(state_dim,action_dim,hidden_dim=120).to(device=device)
    critic_target.load_state_dict(critic.state_dict())
    #actor and critic optimizers
    actor_optimizer = optim.Adam(actor.parameters(),lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(),lr=critic_lr)
    #memory
    memory = DDPG_Memory(capacity=10000)
    # episodes
    total_steps = 0
    actor_loss = 0
    critic_loss = 0
    for i in range(episodes):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        building_1=[]
        building_2=[]
        building_3=[]
        building_4=[]
        building_5=[]
        while not done:
            if total_steps < random_steps:
                action = [([random.uniform(-1,1)]) for _ in range(5)]
            else:
                #add gaussian noise 
                action = actor(torch.flatten(torch.FloatTensor(state).to(device=device)))
                action = action.cpu().detach().numpy() + np.random.normal(0,0.1,size=action_dim).clip(-1,1)
                action = [([i]) for i in action]
            next_state, reward, done, _ = env.step(action)
            steps = steps + 1
            building_1.append(reward[0])
            building_2.append(reward[1])
            building_3.append(reward[2])
            building_4.append(reward[3])
            building_5.append(reward[4])
            score = score + reward.sum()
            action = [i[0] for i in action]
            memory.push(state=torch.FloatTensor(state).flatten(),next_state=torch.FloatTensor(next_state).flatten(),action=torch.FloatTensor(action),reward=torch.FloatTensor(reward).sum(),done=torch.tensor(done))
            state = next_state
            if total_steps >= random_steps and total_steps%update_freq == 0:
                for _ in range(update_freq):
                    #learn
                    samples = memory.sample(batch_size=batch_size)
                    next_states = torch.stack(list(samples.next_state)).to(device=device)
                    states = torch.stack(list(samples.state)).to(device=device)
                    actions = torch.stack(list(samples.action)).to(device=device)
                    dones = torch.stack(list(samples.done)).to(device=device)
                    rewards = torch.stack(list(samples.reward)).to(device=device)
                    # Target Q
                    with torch.no_grad():
                        Q_ = critic_target(next_states,actor_target(next_states)).squeeze(dim=1)
                        Q_target = rewards + gamma * (~dones) * Q_
                    #critic update
                    Q_Value = critic(states,actions).squeeze(dim=1)
                    critic_loss = F.mse_loss(Q_target,Q_Value)
                    critic_optimizer.zero_grad()
                    critic_loss.backward() 
                    critic_optimizer.step()

                    # Freeze crtitic network
                    for param in critic.parameters():
                        param.requires_grad = False

                    #actor update
                    actor_loss = -1 * critic(states,actor(states)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Unfreeze critic networks
                    for params in critic.parameters():
                        params.requires_grad = True

                    # soft target update by polyak average
                    for param_critic,target_param_critic,param_actor,target_param_actor in zip(critic.parameters(),critic_target.parameters(),actor.parameters(),actor_target.parameters()):
                        target_param_critic.data.copy_(tau*param_critic.data + (1-tau)*target_param_critic.data)
                        target_param_actor.data.copy_(tau*param_actor.data + (1-tau)*target_param_actor.data)

        total_steps = total_steps + 1   
        wandb.log({"score":score,"actor_loss":actor_loss,"critic_loss":critic_loss,"Building_Score_1":sum(building_1),"Building_Score_2":sum(building_2),"Building_Score_3":sum(building_3),"Building_Score_4":sum(building_4),"Building_Score_5":sum(building_5)})
        print("Episode:",i,"total_score:",score,"Building_Score_1:",sum(building_1),"Building_Score_2:",sum(building_2),"Building_Score_3:",sum(building_3),"Building_Score_4:",sum(building_4),"Building_Score_5:",sum(building_5))

def train_td3_mlp(env,state_dim,action_dim,actor_lr,critic_lr,gamma,tau,episodes,random_steps,batch_size,device,policy_freq):
    wandb.init(project="CITYCLEAN_RL",name="td3_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}".format(actor_lr,critic_lr,gamma,tau))
    #actor and actor target
    actor = TD3_MLP_ACTOR(state_dim,action_dim,hidden_dim=120).to(device=device)
    actor_target = TD3_MLP_ACTOR(state_dim,action_dim,hidden_dim=120).to(device=device)
    actor_target.load_state_dict(actor.state_dict())
    #critic and critic target
    critic = TD3_MLP_CRITIC(state_dim,action_dim,hidden_dim=120).to(device=device)
    critic_target = TD3_MLP_CRITIC(state_dim,action_dim,hidden_dim=120).to(device=device)
    critic_target.load_state_dict(critic.state_dict())
    #actor and critic optimizers
    actor_optimizer = optim.Adam(actor.parameters(),lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(),lr=critic_lr)
    #memory
    memory = TD3_Memory(capacity=10000)
    # episodes
    total_steps = 0
    actor_loss = 0
    critic_loss = 0
    actor_pointer = 0
    for i in range(episodes):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        building_1=[]
        building_2=[]
        building_3=[]
        building_4=[]
        building_5=[]
        while not done:
            if total_steps < random_steps:
                action = [([random.uniform(-1,1)]) for _ in range(5)]
            else:
                #add gaussian noise 
                action = actor(torch.flatten(torch.FloatTensor(state).to(device=device)))
                action = action.cpu().detach().numpy() + np.random.normal(0,0.1,size=action_dim).clip(-1,1)
                action = [([i]) for i in action]
            next_state, reward, done, _ = env.step(action)
            steps = steps + 1
            building_1.append(reward[0])
            building_2.append(reward[1])
            building_3.append(reward[2])
            building_4.append(reward[3])
            building_5.append(reward[4])
            score = score + reward.sum()
            action = [i[0] for i in action]
            memory.push(state=torch.FloatTensor(state).flatten(),next_state=torch.FloatTensor(next_state).flatten(),action=torch.FloatTensor(action),reward=torch.FloatTensor(reward).sum(),done=torch.tensor(done))
            state = next_state
            if total_steps >= random_steps:
                actor_pointer = actor_pointer + 1
                #learn
                samples = memory.sample(batch_size=batch_size)
                next_states = torch.stack(list(samples.next_state)).to(device=device)
                states = torch.stack(list(samples.state)).to(device=device)
                actions = torch.stack(list(samples.action)).to(device=device)
                dones = torch.stack(list(samples.done)).to(device=device)
                rewards = torch.stack(list(samples.reward)).to(device=device)
                # Target Q
                with torch.no_grad():
                    # target policy smoothing 
                    noise = (torch.rand_like(actions) * 0.2).clamp(-0.5,0.5)
                    next_action = (actor_target(next_states) + noise).clamp(-1,1)
                    # clipped double q learning
                    target_q1,target_q2 = critic_target(next_states,next_action).squeeze(dim=1)
                    target_q = rewards + gamma * (~dones) * torch.min(target_q1,target_q2)
                
                # critic update
                q1,q2 = critic(states,actions).squeeze(dim=1)
                critic_loss = F.mse_loss(q1,target_q1) + F.mse_loss(q2,target_q2)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # actor update
                if actor_pointer % policy_freq == 0:
                    # freeze critic
                    for params in critic.parameters():
                        params.requires_grad = False
                    q1,q2 = critic(states,actor(states))
                    actor_loss = -1 * q1.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                # unfreeze critic
                for params in critic.parameters():
                    params.requires_grad = True

                # soft target update by polyak average
                for param_critic,target_param_critic,param_actor,target_param_actor in zip(critic.parameters(),critic_target.parameters(),actor.parameters(),actor_target.parameters()):
                    target_param_critic.data.copy_(tau*param_critic.data + (1-tau)*target_param_critic.data)
                    target_param_actor.data.copy_(tau*param_actor.data + (1-tau)*target_param_actor.data)
        
        total_steps = total_steps + 1   
        wandb.log({"score":score,"actor_loss":actor_loss,"critic_loss":critic_loss,"Building_Score_1":sum(building_1),"Building_Score_2":sum(building_2),"Building_Score_3":sum(building_3),"Building_Score_4":sum(building_4),"Building_Score_5":sum(building_5)})
        print("Episode:",i,"total_score:",score,"Building_Score_1:",sum(building_1),"Building_Score_2:",sum(building_2),"Building_Score_3:",sum(building_3),"Building_Score_4:",sum(building_4),"Building_Score_5:",sum(building_5))








    



