import os
import numpy as np
import scipy.io as scio
import torch
from DDPG import DDPG
from Prius_model_new import Prius

# Load the trained model
actor_model = torch.load('Checkpoints/source/actor_model.pt')
critic_model = torch.load('Checkpoints/source/critic_model.pt')

# Initialize the Prius class
Prius = Prius()

# Initialize necessary variables
s_dim = 3
a_dim = 1
a_bound = 1
ddpg = DDPG(s_dim, a_dim, a_bound)
ddpg.actor_net.load_state_dict(actor_model)
ddpg.critic_net.load_state_dict(critic_model)

# Load the test data
path = "Data_Standard Driving Cycles/Prius_source_data"
path_list = os.listdir(path)
random_data = np.random.randint(0, len(path_list))
base_data = path_list[random_data]
data = scio.loadmat(path + "/" + base_data)
car_spd_one = data["speed_vector"]
total_milage = np.sum(car_spd_one) / 1000

# Loop over the test data
SOC = 0.65
for i in range(car_spd_one.shape[1] - 1):
    car_spd = car_spd_one[:, i]
    car_a = car_spd_one[:, i+1] - car_spd_one[:, i]
    s = np.zeros(s_dim)
    s[0] = car_spd / 24.1683
    s[1] = (car_a - (-1.6114)) / (1.3034 - (-1.6114))
    s[2] = SOC

    action, _ = ddpg.choose_action(s)
    Eng_pwr_opt = action * 56000

    out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
    SOC = float(out["SOC"])
    
    # Print the cost and other necessary information
    print(f'Step {i}, cost: {cost}, SOC: {SOC}')

