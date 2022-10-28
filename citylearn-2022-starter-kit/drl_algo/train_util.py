# The training code for DDPG,TD3 and SAC
from importlib.resources import path
import os
from re import A
import sys
from typing_extensions import final
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


def get_crtic_from_action(n_buildings, n_agents,observation_space, action, critic_m):
    avg_critic = torch.zeros((observation_space.shape[0], 1), device='cuda')
    for i in range(n_buildings-n_agents+1):
        critic_ = critic_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28), action[:,i:i+n_agents]).view(-1,1,1)
        avg_critic = torch.cat((avg_critic, critic_.view(-1, 1)), dim=1)
    avg_critic = torch.sum(avg_critic, dim=1)
    avg_critic/=(n_buildings-n_agents+1)

    return avg_critic
    

def get_generalized_actions(n_buildings, n_agents, observation_space, actor_m, critic_m):
    # print(n_buildings, n_agents)
    each_action = torch.zeros((observation_space.shape[0],n_buildings), device='cuda').view(-1,1,n_buildings)
    each_critic = torch.zeros((observation_space.shape[0], n_buildings), device='cuda').view(-1,1,n_buildings)
    # print(observation_space.shape)
    avg_critic = torch.zeros((observation_space.shape[0], 1), device='cuda')
    for i in range(n_buildings-n_agents+1):
        # print(observation_space[i:i+n_agents].view(-1, n_buildings*28).shape)
        action = actor_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28)).view(-1,n_agents)
        # print(observation_space[i:i+n_agents].view(-1,1).shape, action.shape)
        critic_ = critic_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28), action).view(-1,1,1)
        avg_critic = torch.cat((avg_critic, critic_.view(-1, 1)), dim=1)
        action = action.view(-1,1,n_agents)
        # print(critic_.shape)
        critic = critic_.repeat((1,1,n_agents))
        # print(critic.shape)
        # for i in range(n_agents):
        #     critic = torch.cat((critic, critic_.copy_()), dim=1)

        # print("ads",critic.shape, action.shape)

        if i!=0:
            before = torch.zeros((observation_space.shape[0],i), device='cuda').view(-1, 1, i)
            action = torch.cat((before, action), dim=2)
            critic = torch.cat((before, critic), dim=2)
            # print(before.shape, action.shape)

        if n_buildings-n_agents-i != 0:
            after = torch.zeros((observation_space.shape[0],n_buildings-n_agents-i), device='cuda').view(-1,1,n_buildings-n_agents-i)
            # print(after.shape, action.shape)
            action = torch.cat((action, after), dim=2)
            critic = torch.cat((critic, after), dim=2)
            # print(after.shape, action.shape)

        # print(action.shape, each_action.shape)

        each_action = torch.cat((each_action, action), dim=1)
        each_critic = torch.cat((each_critic, critic), dim=1)

    # print(each_action)
    # print(each_critic)
    final_action = torch.sum(each_action*each_critic, dim=1)
    norm = torch.sum(each_critic, dim=1)

    avg_critic = torch.sum(avg_critic, dim=1)
    avg_critic/=(n_buildings-n_agents+1)
    final_action/=norm
    

    # print(final_action.shape)
    return final_action, avg_critic


def train_ddpg_mlp(env, observation_total_dim, state_dim,action_dim,actor_lr,critic_lr,gamma,tau,episodes,random_steps,update_freq,batch_size,device, r):
    wandb.init(project="REWARD_SWEEP",name="GENERALIZED_DPPG_reward_{}_ddpg_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}".format(r, actor_lr,critic_lr,gamma,tau),entity="cleancity_challenge_rl")
    #actor and actor target
    new_best = 10000
    actor = DDPG_MLP_ACTOR(state_dim,action_dim,hidden_dim=70).to(device=device)
    actor_target = DDPG_MLP_ACTOR(state_dim,action_dim,hidden_dim=70).to(device=device)
    actor_target.load_state_dict(actor.state_dict())
    #critic and critic target
    critic = DDPG_MLP_CRITIC(state_dim,action_dim,hidden_dim=70).to(device=device)
    critic_target = DDPG_MLP_CRITIC(state_dim,action_dim,hidden_dim=70).to(device=device)
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

        # random_steps = 1
        while not done:
            if total_steps < random_steps:
                action = [([random.uniform(-1,1)]) for _ in range(5)]
            else:
                #add gaussian noise 
                # action = actor(torch.flatten(torch.FloatTensor(state).to(device=device)))
                # print(torch.FloatTensor(state).shape)
                action, _ = get_generalized_actions(observation_total_dim, action_dim, torch.FloatTensor(state).to(device=device).view(1,observation_total_dim, 28), actor, critic)
                action = action.view(-1)
                action = (action.cpu().detach().numpy() + np.random.normal(scale=0.3,size=5)).clip(-1,1)
                action = [([i]) for i in action]
            next_state, reward, done, _ = env.step(action)
            if steps == 500:
                done = True
            steps = steps + 1
            building_1.append(reward[0])
            building_2.append(reward[1])
            building_3.append(reward[2])
            building_4.append(reward[3])
            building_5.append(reward[4])
            score = score + reward.sum()
            action = [i[0] for i in action]
            memory.push(state=torch.FloatTensor(state),next_state=torch.FloatTensor(next_state),action=torch.FloatTensor(action),reward=torch.FloatTensor(reward).sum(),done=torch.tensor(done))
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
                        # print(next_states.shape)
                        action_, Q_ = get_generalized_actions(observation_total_dim, action_dim, next_states, actor_target, critic_target)
                        # print(action_.shape)
                        # Q_ = critic_target(next_states.view(-1, ), action_).squeeze(dim=1)
                        Q_target = rewards + gamma * (~dones) * Q_
                    #critic update
                    # action_, Q_Value = get_generalized_actions(observation_total_dim, action_dim, states, actor_target, critic)
                    Q_Value = get_crtic_from_action(observation_total_dim, action_dim, states, actions, critic)
                    # Q_Value = critic(states.flatten(),actions).squeeze(dim=1)
                    critic_loss = F.mse_loss(Q_target,Q_Value)
                    critic_optimizer.zero_grad()
                    critic_loss.backward() 
                    critic_optimizer.step()

                    # Freeze crtitic network
                    for param in critic.parameters():
                        param.requires_grad = False

                    #actor update
                    action__, critic__ = get_generalized_actions(observation_total_dim, action_dim, states, actor, critic)
                    actor_loss = -1 * critic__.mean()
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
        metrics_t = env.evaluate()


        wandb.log({"score":score,"actor_loss":actor_loss,"critic_loss":critic_loss,"Building_Score_1":sum(building_1),"Building_Score_2":sum(building_2),"Building_Score_3":sum(building_3),"Building_Score_4":sum(building_4),"Building_Score_5":sum(building_5)})
        print("Episode:",i,"total_score:",score,"Building_Score_1:",sum(building_1),"Building_Score_2:",sum(building_2),"Building_Score_3:",sum(building_3),"Building_Score_4:",sum(building_4),"Building_Score_5:",sum(building_5), "Price cost:",metrics_t[0], "Emission cost", metrics_t[1], "metrics", sum(metrics_t), end="\n\n")
        print("CURRENT BEST TOTAL SCORE: ", new_best)
        # print("ACTION: ",action, end = "\n")
        # print("REWARD: ",reward, end = "\n\n")
        if new_best>sum(metrics_t):
            torch.save(actor.state_dict(),"{}general_ddpg-actor_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}.pth".format(path_checkpoint,actor_lr,critic_lr,gamma,tau))
            torch.save(critic.state_dict(),"{}general_ddpg-critic_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}.pth".format(path_checkpoint,actor_lr,critic_lr,gamma,tau))
            new_best = sum(metrics_t)


def train_td3_mlp(env,state_dim,action_dim,actor_lr,critic_lr,gamma,tau,episodes,random_steps,batch_size,device,policy_freq,r):
    wandb.init(project="REWARD_SWEEP",name="reward_{}_td3_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}".format(r,actor_lr,critic_lr,gamma,tau),entity="cleancity_challenge_rl")
    #actor and actor target
    actor = TD3_MLP_ACTOR(state_dim,action_dim,hidden_dim=70).to(device=device)
    actor_target = TD3_MLP_ACTOR(state_dim,action_dim,hidden_dim=70).to(device=device)
    actor_target.load_state_dict(actor.state_dict())
    #critic and critic target
    critic = TD3_MLP_CRITIC(state_dim,action_dim,hidden_dim=70).to(device=device)
    critic_target = TD3_MLP_CRITIC(state_dim,action_dim,hidden_dim=70).to(device=device)
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
                action = (action.cpu().detach().numpy() + np.random.normal(scale=0.3,size=action_dim)).clip(-1,1)
                action = [([i]) for i in action]
            next_state, reward, done, _ = env.step(action)
            if steps == 500:
                 done = True
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
                    target_q1,target_q2 = critic_target(next_states,next_action)
                    target_q1,target_q2 = target_q1.squeeze(dim=1),target_q2.squeeze(dim=1)
                    target_q = rewards + gamma * (~dones) * torch.min(target_q1,target_q2)
                
                # critic update
                q1,q2 = critic(states,actions)
                q1, q2 = q1.squeeze(dim=1), q2.squeeze(dim=1)
                critic_loss = F.mse_loss(q1,target_q) + F.mse_loss(q2,target_q)
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

        metrics_t = env.evaluate()

        metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}

        wandb.log({"score":score,"actor_loss":actor_loss,"critic_loss":critic_loss,"Building_Score_1":sum(building_1),"Building_Score_2":sum(building_2),"Building_Score_3":sum(building_3),"Building_Score_4":sum(building_4),"Building_Score_5":sum(building_5), "metric":sum(metrics_t), "Price cost":metrics_t[0], "Emmision cost":metrics_t[1]})


        print("Episode:",i,"total_score:",score,"Building_Score_1:",sum(building_1),"Building_Score_2:",sum(building_2),"Building_Score_3:",sum(building_3),"Building_Score_4:",sum(building_4),"Building_Score_5:",sum(building_5), "Price cost:",metrics_t[0], "Emission cost", metrics_t[1], "metrics", sum(metrics_t))
        torch.save(actor.state_dict(),"{}td3-actor_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}.pth".format(path_checkpoint,actor_lr,critic_lr,gamma,tau))
        torch.save(critic.state_dict(),"{}td3-critic_mlp_actor-lr:{}_critic-lr:{}_gamma:{}_tau:{}.pth".format(path_checkpoint,actor_lr,critic_lr,gamma,tau))





