
from os import stat
from drl_algo.models import DDPG_MLP_ACTOR,DDPG_MLP_CRITIC,TD3_MLP_ACTOR,TD3_MLP_CRITIC
import torch
from citylearn.citylearn import CityLearnEnv

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

env = CityLearnEnv(schema=Constants.schema_path)
actor = TD3_MLP_ACTOR(env.observation_space[0].shape[0]*5,env.action_space[0].shape[0]*5,hidden_dim=120)
actor.load_state_dict(torch.load("/home/g0kul6/g0kul6/cityclean-rl/checkpoint/td3-actor_mlp_actor-lr_0.0003_critic-lr_0.0003_gamma_0.99_tau_0.05.pth"))
state = env.reset()
score = 0
step = 0
done = False
while not done:
    action = actor(torch.flatten(torch.FloatTensor(state)))
    action = [([i]) for i in action]
    next_state, reward, done, _ = env.step(action)
    step = step + 1
    score = score + reward
    metrics_t = env.evaluate()
    metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
    if step%100 == 0:
        print(metrics)
        print("Step:",step,"Score:",sum(score))
        print(action)