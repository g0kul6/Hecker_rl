from typing import List
import numpy as np
import os
###########################################################################
#####                Specify your reward function here                #####
###########################################################################
E_CONST = np.e**-4
key = os.listdir()
for i in key:
    if i[:3] == "KEY":
        key = i[3:]
print(key)

class keys_store:
    reward_key :int = key

def custom_rewards(x, which=0):
    func = {0: lambda x: x*-1,
            1: lambda x: 1/(x+E_CONST),
            2: lambda x: 1/(1 + np.exp(x)),
            # 3: lambda x: np.asarray([-math.log(i) if i>0 else math.log(-i) for i in x])
            }

    return func[which](x)

def get_reward(electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float], agent_ids: List[int]) -> List[float]:
        """CityLearn Challenge user reward calculation.

        Parameters
        ----------
        electricity_consumption: List[float]
            List of each building's/total district electricity consumption in [kWh].
        carbon_emission: List[float]
            List of each building's/total district carbon emissions in [kg_co2].
        electricity_price: List[float]
            List of each building's/total district electricity price in [$].
        agent_ids: List[int]
            List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.

        Returns
        -------
        rewards: List[float]
            Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings) 
            or = number of buildings (independent agent for each building).
        """

        # *********** BEGIN EDIT ***********
        # Replace with custom reward calculation
        carbon_emission = np.array(carbon_emission).clip(min=0)
        electricity_price = np.array(electricity_price).clip(min=0)


        reward = custom_rewards(carbon_emission + electricity_price,int(keys_store.reward_key))
        # ************** END ***************
        return reward