from os import stat
from random import sample
from model import PPO_Actor, PPO_Critic
from memory import PPO_Memory, Transition_PPO
import torch 
import torch.optim as optim 
import torch.nn.functional as F 
import numpy as np

def get_crtic_from_action(n_buildings, n_agents,observation_space, action, critic_m):
    avg_critic = torch.zeros((observation_space.shape[0], 1), device='cuda')
    for i in range(n_buildings-n_agents+1):
        critic_ = critic_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28)).view(-1,1,1)
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
        action = actor_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28)).sample().view(-1,n_agents)
        # print(observation_space[i:i+n_agents].view(-1,1).shape, action.shape)
        critic_ = critic_m(observation_space[:,i:i+n_agents].view(-1, n_agents*28)).view(-1,1,1)
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

def train_ppo(env,num_agents,actor_lr,critic_lr,gamma,epsilon,episodes,capacity,batch_size,device):
    obs = env.reset()
    num_buildings =  obs.shape[0]
    actor = PPO_Actor(obs_dim=env.observation_space[0].shape[0]*num_agents,act_dim=num_agents).to(device=device)
    critic = PPO_Critic(obs_dim=env.observation_space[0].shape[0]*num_agents).to(device=device)
    optimizer_actor = optim.Adam(params=actor.parameters(),lr = actor_lr)
    optimizer_critic = optim.Adam(params=critic.parameters(),lr = critic_lr)
    replayBuffer = PPO_Memory(capacity=capacity)
    
    for episode in episodes : 
        state = env.reset()
        done = False
        episode_reward = 0
        reward = 0
        while not done : 
            action, _ = get_generalized_actions(n_buildings=num_buildings,num_agents=num_agents,observation_space=torch.FloatTensor(state).to(device).view(1,num_buildings,28),actor_m=actor,critic_m=critic)
            action = action.view(-1)
            action = action.cpu().detach().numpy()
            action = [[i] for i  in action]
            next_state, reward, done , _ = env.step(action)
            replayBuffer.push(torch.FloatTensor(state),torch.FloatTensor(next_state),torch.FloatTensor(action),torch.FloatTensor(reward).sum(),torch.tensor(done))
            state = next_state

            if len(replayBuffer)>batch_size:
                samples = replayBuffer.sample(batch_size=batch_size)
                next_states = torch.stack(list(samples.next_state)).to(device=device)
                states = torch.stack(list(samples.state)).to(device=device)
                actions = torch.stack(list(samples.action)).to(device=device)
                dones = torch.stack(list(samples.done)).to(device=device)
                # rewards = torch.stack(list(samples.reward)).to(device=device)
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(sample.rewards), reversed(sample.is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + gamma * discounted_reward
                    rewards.insert(0, discounted_reward)

                # Normalize rewards
                rewards = torch.tensor(rewards).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            

            
                


