import argparse
from train import train_ppo
import os 
from citylearn.citylearn import CityLearnEnv
import numpy as np 
import torch 

class Constants: 
    num_episodes = 3
    schema_path = '/media/joy/DATA/Hecker_rl/citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/schema.json'
    seed = 123456
    gamma = 0.99
    epsilon = 0.2
    capacity = 1000

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

env = CityLearnEnv(schema=Constants.schema_path)

parser = argparse.ArgumentParser(description='Enter train loop args : ')
# Algo 
parser.add_argument('--algo',type=str, default='ppo', required=True)
# num_agents
parser.add_argument("--num_agents", type=int, deafult=3,required=True)
# Actor_lr 
parser.add_argument('--actor_lr', type=float, default=1e-3, required=True)
# Critic_lr
parser.add_argument('--critic_lr',type=float, default=1e-3, required=True)
# Device 
parser.add_argument('--device', type=str , default="cpu" , required=True)
# Num_episodes 
parser.add_argument('--episodes', type=int , default=1000, required=True)
# batch_size
parser.add_argument("--batch_size",type=int,default=64,required=True)
parser.add_argument("--reward_key",type=int)
args = parser.parse_args()

os.mkdir("KEY" + str(args.reward_key))
os.rmdir("KEY" + str(args.reward_key))
env.seed(Constants.seed)
torch.manual_seed(Constants.seed)
np.random.seed(Constants.seed)

if args.algo == "ppo":
    train_ppo(env,num_agents=args.num_agents,actor_lr=args.actor_lr,critic_lr=args.critic_lr,gamma=Constants.gamma, epsilon=Constants.epsilon,episodes=args.episodes,capacity=Constants.capacity,batch_size=args.batch_size,device=args.device)
