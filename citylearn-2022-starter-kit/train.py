# file which gets different hyperparameters to train the models 

from drl_algo.train_util import train_ddpg_mlp,train_td3_mlp
from drl_algo.config import actor_lr,critic_lr,gamma,tau,batch_size
from citylearn.citylearn import CityLearnEnv
import torch
import numpy as np
class Constants:
    episodes = 3
    schema_path = 'citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/schema.json'

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



import argparse
parser = argparse.ArgumentParser()
#algo
parser.add_argument("--algo",type=str,default="ddpg",required=True)
#actor lr
parser.add_argument("--actor_lr",type=float,default=actor_lr,required=True)
#critic lr
parser.add_argument("--critic_lr",type=float,default=critic_lr,required=True)
#gamma
parser.add_argument("--gamma",type=float,default=gamma,required=True)
#tau
parser.add_argument("--tau",type=float,default=tau,required=True)
#device
parser.add_argument("--device",type=str,default="cuda",required=True)
#episodes
parser.add_argument("--epochs",type=int,default=1000,required=True)
parser.add_argument('--reward_key', type=int)
args = parser.parse_args()

import os
os.mkdir("KEY"+str(args.reward_key))
env = CityLearnEnv(schema=Constants.schema_path)
os.rmdir("KEY"+str(args.reward_key))

env.seed(123456)

torch.manual_seed(123456)
np.random.seed(123456)

if args.algo == "ddpg":
    train_ddpg_mlp(env,5, 28*3,3,actor_lr=args.actor_lr,critic_lr=args.critic_lr,tau=args.tau,
                batch_size=batch_size,device=args.device,random_steps=50,episodes=args.epochs,update_freq=7,gamma=args.gamma, r=args.reward_key)
elif args.algo == "td3":
    train_td3_mlp(env=env,state_dim=env.observation_space[0].shape[0]*5,action_dim=env.action_space[0].shape[0]*5,actor_lr=args.actor_lr,critic_lr=args.critic_lr,tau=args.tau,
                batch_size=batch_size,device=args.device,random_steps=50,episodes=args.epochs,gamma=args.gamma,policy_freq=2, r=args.reward_key)
elif args.algo == "sac":
    pass
