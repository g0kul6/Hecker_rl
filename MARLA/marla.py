import os
import sys
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from citylearn.citylearn import CityLearnEnv
from memory import DDPG_Memory



LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


path_dir = os.path.abspath(os.getcwd())
path_checkpoint = path_dir + "/checkpoint/"


class Constants:
    episodes = 3
    schema_path = 'citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/schema.json'

class ARGs:
    reward_key = 0
    device = 'cuda'
    epochs = 1000
    actor_lr = 1e-4
    critic_lr = 1e-4
    gamma = 0.99
    batch_size = 256
    tau = 0.05
    state_dim = 28
    action_dim = 1
    critic_hidden_dim = 32
    actor_hidden_dim = 32
    extractor_hidden_dim = 32
    attn_hidden_dim = 32
    n_agents = 5
    n_heads = 2
    device = 'cuda'
    update_freq = 7
    random_steps = 50
    max_steps = 500


args = ARGs()

os.mkdir("KEY"+str(args.reward_key))
env = CityLearnEnv(schema=Constants.schema_path)
os.rmdir("KEY"+str(args.reward_key))

env.seed(123456)

torch.manual_seed(123456)
np.random.seed(123456)



def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict


import random
# from tqdm.auto import tqdm



class DDPG_MLP_ACTOR(nn.Module):
    def __init__(self,state_dim,action_dim,actor_hidden_dim):
        super().__init__()
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(state_dim,actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim,actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim,action_dim),
            nn.Tanh()
        )
    
    def forward(self,s):
        action = self.network_actor(s)
        return action

class DDPG_MLP_CRITIC(nn.Module):
    def __init__(self,state_dim,action_dim,critic_hidden_dim):
        super().__init__()
        #critic
        self.network_critic = nn.Sequential(
            nn.Linear(state_dim+action_dim,critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim,critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim,1)
        )
    
    def forward(self,s,a):
        q_value = self.network_critic(torch.cat([s, a], 1))
        return q_value


class STATE_ATTN_EXTRACTOR(nn.Module):
    def __init__(self, state_dim, extractor_hidden_dim = 32 , attn_hidden_dim = 32, n_agents = 5, n_heads = 2):
        super().__init__()
        self.attention = nn.MultiheadAttention(attn_hidden_dim, n_heads, batch_first=True)  

    
    def forward(self, x):
        
        attn_output, attn_output_weights = self.attention (x, x, x)
        return attn_output



class MARLA():
    def __init__(self, state_dim, action_dim, critic_hidden_dim, actor_hidden_dim, extractor_hidden_dim = 32 , attn_hidden_dim = 32, n_agents = 5, n_heads = 2, device = 'cuda', update_freq = 7, random_steps = 50, max_steps=500):
        self.critic = [DDPG_MLP_CRITIC(attn_hidden_dim, action_dim, critic_hidden_dim).to(device=device) for i in range(n_agents)]
        self.critic_target = [DDPG_MLP_CRITIC(attn_hidden_dim, action_dim, critic_hidden_dim).to(device=device) for i in range(n_agents)]

        self.actor = [DDPG_MLP_ACTOR(attn_hidden_dim, action_dim, actor_hidden_dim).to(device=device) for i in range(n_agents)]
        self.actor_target = [DDPG_MLP_ACTOR(attn_hidden_dim, action_dim, actor_hidden_dim).to(device=device) for i in range(n_agents)]


        for i in range(n_agents):
            self.critic_target[i].load_state_dict(self.critic[i].state_dict())
            self.actor_target[i].load_state_dict(self.actor[i].state_dict())

        self.attn_extractor = STATE_ATTN_EXTRACTOR(state_dim, extractor_hidden_dim, attn_hidden_dim, n_agents, n_heads).to(device)
        self.extractor = [nn.Sequential(
            nn.Linear(state_dim, extractor_hidden_dim),
            nn.ReLU(),
            nn.Linear(extractor_hidden_dim, attn_hidden_dim),

        ).to(device) for i in range(n_agents)]    

        self.memory = DDPG_Memory(capacity=10000)
        self.n_agents = n_agents
        self.device = device
        self.random_steps = random_steps
        self.update_freq = update_freq
        self.max_steps = max_steps

        acotor_params = []
        for i in range(n_agents):
            acotor_params+=self.actor[i].parameters()

        critic_params = []
        for i in range(n_agents):
            critic_params+=self.critic[i].parameters()

        extractor_params = []
        for i in range(n_agents):
            extractor_params+=self.extractor[i].parameters()


        self.actor_optimizer = optim.Adam(acotor_params,lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params,lr=args.critic_lr)
        self.attn_extractor_optimizer = torch.optim.Adam(self.attn_extractor.parameters(), lr=0.001)
        self.extractor_optimizer = torch.optim.Adam(extractor_params, lr=0.001)

    def getFeatures(self, states):
        return torch.concat([self.extractor[i](torch.FloatTensor(states[i]).to(self.device)).unsqueeze(0).to(self.device) for i in range(self.n_agents)], 0) #[batch, n_agents, n_feature]

    def getBatchFeatures(self, mult_states):
        mult_states = np.asarray(mult_states)
        return torch.concat([self.extractor[i](torch.FloatTensor(mult_states[:,i]).to(self.device)).to(self.device).unsqueeze(1) for i in range(self.n_agents)], 1)

    def train_step(self, env, total_steps):
        state = np.asarray(env.reset())
        state = state/state.max(axis=0)
        # state = state.tolist()
        score = 0
        done = False
        steps = 0
        building_1=[]
        building_2=[]
        building_3=[]
        building_4=[]
        building_5=[]
        actor_loss = 0
        critic_loss = 0
        # pbar = tqdm(total=self.max_steps)
        while not done:
            # pbar.update(1)
            action = []
            if total_steps < self.random_steps:
                action = [([random.uniform(-1,1)]) for _ in range(5)]
            else:
                #add gaussian noise 
                features = self.getFeatures(state)
                # print(features.shape)
                attn = self.attn_extractor(features.to(self.device))

                for nth_agent in range(self.n_agents):
                    action.append(([(self.actor[nth_agent](attn[nth_agent]).cpu().detach().numpy() + np.random.normal(scale=0.3,size=1)).clip(-1,1)][0].tolist()))

            # print(action)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray(next_state)
            next_state = next_state/next_state.max(axis=0)
            # next_state = next_state.tolist()
            if steps == self.max_steps:
                done = True
            steps = steps + 1

            building_1.append(reward[0])
            building_2.append(reward[1])
            building_3.append(reward[2])
            building_4.append(reward[3])
            building_5.append(reward[4])
            score = score + reward.sum()
            action = [i[0] for i in action]

            self.memory.push(state,next_state=next_state,action=torch.FloatTensor(action),reward=torch.FloatTensor(reward).sum(),done=torch.tensor(done))
            state = next_state

            if total_steps >= self.random_steps and total_steps%self.update_freq == 0:
                for _ in range(self.update_freq):
                    #learn
                    samples = self.memory.sample(batch_size=args.batch_size)
                    next_states = list(samples.next_state)
                    states = list(samples.state)
                    actions = torch.stack(list(samples.action)).to(device=self.device)
                    dones = torch.stack(list(samples.done)).to(device=self.device)
                    rewards = torch.stack(list(samples.reward)).to(device=self.device)
                    # Target Q
                    Q_ = []
                    Q_target = []
                    with torch.no_grad():
                        features = self.getBatchFeatures(next_states)
                        attn = self.attn_extractor(features.to(self.device))

                        Q_ = [self.critic_target[i](attn[:,i],self.actor_target[i](attn[:,i])).squeeze(dim=1) for i in range(self.n_agents)]
                        Q_target = torch.cat([(rewards+ args.gamma * (~dones) * Q_[i]).unsqueeze(1) for i in range(self.n_agents)],1)

                    #critic update
                    i=0
                    # print(self.critic[i](attn[:,i],actions[:,i].unsqueeze(0)).squeeze(dim=1).shape)
                    Q_Value = torch.cat([self.critic[i](attn[:,i],actions[:,i].unsqueeze(1)) for i in range(self.n_agents)],1)

                    critic_loss = F.mse_loss(Q_target,Q_Value)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward() 
                    self.critic_optimizer.step()
                    # Freeze crtitic network
                    for i in range(self.n_agents):
                        for param in self.critic[i].parameters():
                            param.requires_grad = False
                    #actor update

                    features = self.getFeatures(next_states)
                    attn = self.attn_extractor(features.to(self.device))

                    actor_loss = -1 * torch.cat([self.critic[i](attn[:,i],self.actor[i](attn[:,i])) for i in range(self.n_agents)],0).mean()
                    self.actor_optimizer.zero_grad()
                    self.extractor_optimizer.zero_grad()
                    self.attn_extractor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.extractor_optimizer.step()
                    self.attn_extractor_optimizer.step()
                    # Unfreeze critic networks
                    for i in range(self.n_agents):
                        for param in self.critic[i].parameters():
                            param.requires_grad = True

                    # soft target update by polyak average
                    for i in range(self.n_agents):
                        for param_critic,target_param_critic,param_actor,target_param_actor in zip(self.critic[i].parameters(),self.critic_target[i].parameters(),self.actor[i].parameters(),self.actor_target[i].parameters()):
                            target_param_critic.data.copy_(args.tau*param_critic.data + (1-args.tau)*target_param_critic.data)
                            target_param_actor.data.copy_(args.tau*param_actor.data + (1-args.tau)*target_param_actor.data)

        # pbar.close()
        metrics_t = env.evaluate()
        return score, actor_loss, critic_loss, building_1, building_2, building_3, building_4, building_5, metrics_t


marla_env = MARLA(state_dim = args.state_dim, action_dim = args.action_dim, critic_hidden_dim = args.critic_hidden_dim, actor_hidden_dim = args.actor_hidden_dim, extractor_hidden_dim = args.extractor_hidden_dim , attn_hidden_dim = args.attn_hidden_dim, n_agents = args.n_agents, n_heads = args.n_heads, device = args.device, update_freq = args.update_freq, random_steps = args.random_steps, max_steps = args.max_steps)
total_steps=0
for i in range(1000):
    wandb.init(project="MARLA",name="MARLA-lr:{}_critic-lr:{}_gamma:{}_tau:{}".format(args.actor_lr,args.critic_lr,args.gamma,args.tau),entity="cleancity_challenge_rl")
    wandb.config = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
    
    score, actor_loss, critic_loss, building_1, building_2, building_3, building_4, building_5, metrics_t = marla_env.train_step(env, total_steps)
    total_steps +=1

    wandb.log({"score":score,"actor_loss":actor_loss,"critic_loss":critic_loss,"Building_Score_1":sum(building_1),"Building_Score_2":sum(building_2),"Building_Score_3":sum(building_3),"Building_Score_4":sum(building_4),"Building_Score_5":sum(building_5), "metric":sum(metrics_t), "Price cost":metrics_t[0], "Emmision cost":metrics_t[1]})
    print("Episode:",i,"total_score:",score,"Building_Score_1:",sum(building_1),"Building_Score_2:",sum(building_2),"Building_Score_3:",sum(building_3),"Building_Score_4:",sum(building_4),"Building_Score_5:",sum(building_5), "Price cost:",metrics_t[0], "Emission cost", metrics_t[1], "metrics", sum(metrics_t))