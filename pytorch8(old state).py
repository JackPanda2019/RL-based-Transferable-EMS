import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as scio
from Prius_model_new import Prius_model
from torch.utils.tensorboard import SummaryWriter
import datetime
import pandas as pd
import traci

end_time = 2101
sumoCmd = ['sumo', '--random', '-c', 'simulation_config.sumocfg','--no-step-log']
#sumoCmd = ['sumo', '-c', 'simulation_config.sumocfg','--no-step-log']

device = torch.device("cuda")
now = datetime.datetime.now()
time_string1 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____1"
#print(time_string1)
time_string2 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____2"
#print(time_string2)
time_string3 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____3"
#print(time_string3)
time_string4 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____4"
#print(time_string4)
time_string5 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____5"
#print(time_string5)
time_string6 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____6"
#print(time_string6)
time_string7 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____7"
#print(time_string7)
time_string8 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____8"
#print(time_string8)
time_string9 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____9"
#print(time_string9)
time_string10 = now.strftime("%Y-%m-%d_%H-%M-%S")+"_____10"
#print(time_string10)



MAX_EPISODES = 150
LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64
RENDER = False

first_noise = "gs"
second_noise = "None"

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, output_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action = torch.sigmoid(self.fc4(x))
        scaled_action = action * self.action_bound
        return scaled_action

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, writer):
        self.writer = writer
        self.global_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)
        self.memory = Memory(MEMORY_CAPACITY)
        self.td_loss = nn.MSELoss()
        self.ou_noise_prev = 0

    def NormalActionNoise(self, mu, sigma):
        return (torch.randn(1) * sigma + mu).item()

    def OrnsteinUhlenbeckActionNoise(self, mu, sigma, theta=0.15, dt=1e-2):
        ou_noise = (
            self.ou_noise_prev
            + theta * (mu - self.ou_noise_prev) * dt
            + sigma
            * torch.sqrt(torch.tensor(dt))
            * (torch.randn(size=(1,)) * sigma + mu)
        )
        self.ou_noise_prev = ou_noise
        return ou_noise.item()

    def choose_action(
        self, state, loop, param_noise_scale, param_noise=False, action_noise_type=None
    ):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            output_no_noise = (
                self.actor(state).squeeze().cpu().numpy()
            )  # Forward pass without noise

        # Add parameter noise
        if param_noise and (loop == 0):
            with torch.no_grad():
                for param in self.actor.parameters():
                    param.add_(torch.randn_like(param).to(device) * param_noise_scale)
            output = self.actor(state).squeeze().cpu().numpy()
        else:
            output = output_no_noise

        # Add action noise
        if action_noise_type == "gs":
            output_noise =  np.random.rand()  # Gaussian noise
        elif action_noise_type == "ou":
            output_noise = output + self.OrnsteinUhlenbeckActionNoise(
                0, 0.15, 0.2, 0.01
            )  # OU noise
        elif action_noise_type == "None":
            output_noise = output

        return output_noise, output_no_noise

    def learn(self):
        self.actor.train()
        self.critic.train()
        tree_index, bt, ISWeight = self.memory.sample(BATCH_SIZE)
        bs = torch.FloatTensor(bt[:, : self.state_dim]).to(device)
        ba = torch.FloatTensor(
            bt[:, self.state_dim : self.state_dim + self.action_dim]
        ).to(device)
        br = torch.FloatTensor(bt[:, -self.state_dim - 1 : -self.state_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.state_dim :]).to(device)

        with torch.no_grad():
            next_action = self.actor_target(bs_)
            q_next = self.critic_target(bs_, next_action)
            q_target = br + GAMMA * q_next
        q_value = self.critic(bs, ba)

        td_error = self.td_loss(q_value, q_target)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
        self.writer.add_scalar("Critic TD error", td_error.item(), self.global_step)
        a_loss = -self.critic(bs, self.actor(bs)).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        self.writer.add_scalar("Actor loss", a_loss.item(), self.global_step)
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        abs_td_error = torch.abs(q_target - q_value).detach().cpu().numpy()
        self.memory.batch_update(tree_index, abs_td_error)
        self.global_step += 1

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state))
        self.memory.store(transition)

    def savemodel(self, time_string):
        actor_model_path = f"Checkpoints/source/actor_model{time_string}.pt"
        critic_model_path = f"Checkpoints/source/critic_model{time_string}.pt"
        torch.save(self.actor.state_dict(), actor_model_path)
        torch.save(self.critic.state_dict(), critic_model_path)
    def loadmodel(self,time_string):
        actor_model_path = f"Checkpoints/source/actor_model{time_string}.pt"
        critic_model_path = f"Checkpoints/source/critic_model{time_string}.pt"
        self.actor.load_state_dict(torch.load(actor_model_path))
        self.actor.eval()

        self.critic.load_state_dict(torch.load(critic_model_path))
        self.critic.eval()

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity :])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = (
            np.empty((n,), dtype=np.int32),
            np.empty((n, self.tree.data[0].size)),
            np.empty((n, 1)),
        )
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min(
            [1.0, self.beta + self.beta_increment_per_sampling]
        )  # max = 1

        min_prob = (
            np.min(self.tree.tree[-self.tree.capacity :]) / self.tree.total_p
        )  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while (
            tree_idx != 0
        ):  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

def run_ddpg():
    log_dir = os.path.join("runs", time_string)
    writer = SummaryWriter(log_dir=log_dir)
    s_dim = 3
    a_dim = 1
    a_bound = 1
    ddpg = DDPG(s_dim, a_dim, a_bound, writer)
    total_step = 0
    step_episode = 0
    cost_Engine_list = []
    cost_all_list = []
    mean_reward_list = []
    std_reward_list = []
    SOC_final_list = []

    Prius = Prius_model()
    for i in range(MAX_EPISODES):

        traci.start(sumoCmd)
        for sim_step in range(601):
            traci.simulationStep()

        #path = "Data_Standard Driving Cycles/Prius_source_data"
        #path_list = os.listdir(path)
        #random_data = np.random.randint(0, len(path_list))
        #base_data = path_list[random_data]
        #data = scio.loadmat(path + "/" + base_data)
        #car_spd_one = data["speed_vector"]
        #total_milage = np.sum(car_spd_one) / 1000

        SOC = 0.65
        SOC_origin = SOC
        ep_reward = 0
        ep_reward_all = 0
        step_episode += 1
        SOC_data = []
        action_data = []
        P_req_list = []
        P_out_list = []
        Eng_spd_list = []
        Eng_trq_list = []
        Eng_pwr_list = []
        Eng_pwr_opt_list = []
        Gen_spd_list = []
        Gen_trq_list = []
        Gen_pwr_list = []
        Mot_spd_list = []
        Mot_trq_list = []
        Mot_pwr_list = []
        Batt_pwr_list = []
        inf_batt_list = []
        inf_batt_one_list = []
        Reward_list = []
        Reward_list_all = []
        T_list = []
        Mot_eta_list = []
        Gen_eta_list = []
        #car_spd = car_spd_one[:, 0]
        #print("total_step:",total_step,"----------")
        #current_time = traci.simulation.getTime()
        #print("current_time------------------:",current_time)
        car_spd = traci.vehicle.getSpeed('agent')
        #print("car_spd------------------:",car_spd)
        car_a = traci.vehicle.getAcceleration('agent')
        #print("car_a------------------:",car_a)
        road_id = traci.vehicle.getRoadID('agent')
        #print("road_id------------------:",road_id)
        lane_id = road_id + '_0'
        #print("lane_id------------------:",lane_id)
        occupancy = traci.lane.getLastStepOccupancy(lane_id)
        #print("occupancy------------------:",occupancy)
        next_tls_info = traci.vehicle.getNextTLS('agent')
        tls_id, tls_index, dist, tls_state = next_tls_info[0]
        #print("tls_id------------------:",tls_id)
        #print("tls_index------------------:",tls_index)
        #print("dist------------------:",dist)
        #print("tls_state------------------:",tls_state)
        if tls_state == "r" or tls_state == "y":
            tls_state = 1
        else:
            tls_state = 0

        #car_a = car_spd_one[:, 0] - 0
        s = np.zeros(s_dim)
        s[0] = car_spd / 17.67
        s[1] = (car_a - (-4.5)) / (2.59 - (-4.5))
        s[2] = SOC
        #s[3] = occupancy
        #s[4] = tls_state
        #print("s------------------:",s)
        if total_step > MEMORY_CAPACITY:
            action_noise_type = second_noise
        else:
            action_noise_type = first_noise
        param_noise_scale = 0

        while traci.simulation.getTime() < end_time:
            #print(str(i) + " ---> " + str(j) + "/", car_spd_one.shape[1])
            action, action_no_noise = ddpg.choose_action(
                s, 10, param_noise_scale, False, action_noise_type
            )

            action_data.append(action)
            #print("action------------------------: ", action)
            #a = np.clip(action, 0, 1)
            a = action
            Eng_pwr_opt = a * 56000

            out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
            #P_req_list.append(float(out["P_req"]))
            #P_out_list.append(float(out["P_out"]))
            #Eng_spd_list.append(float(out["Eng_spd"]))
            #Eng_trq_list.append(float(out["Eng_trq"]))
            #Eng_pwr_list.append(float(out["Eng_pwr"]))
            #Eng_pwr_opt_list.append(float(out["Eng_pwr_opt"]))
            #Mot_spd_list.append(float(out["Mot_spd"]))
            #Mot_trq_list.append(float(out["Mot_trq"]))
            #Mot_pwr_list.append(float(out["Mot_pwr"]))
            #Gen_spd_list.append(float(out["Gen_spd"]))
            #Gen_trq_list.append(float(out["Gen_trq"]))
            #Gen_pwr_list.append(float(out["Gen_pwr"]))
            #Batt_pwr_list.append(float(out["Batt_pwr"]))
            #inf_batt_list.append(int(out["inf_batt"]))
            #inf_batt_one_list.append(int(out["inf_batt_one"]))
            #Mot_eta_list.append(float(out["Mot_eta"]))
            #Gen_eta_list.append(float(out["Gen_eta"]))
            #T_list.append(float(out["T"]))
            SOC_new = float(out["SOC"])
            SOC_data.append(SOC_new)
            cost = float(cost)
            r = -cost
            ep_reward += r
            Reward_list.append(r)

            if SOC_new < 0.6 :
                r = -((350 * ((0.6 - SOC_new) ** 2)) + cost)


            
            traci.simulationStep()

            car_spd = traci.vehicle.getSpeed('agent')
            car_a = traci.vehicle.getAcceleration('agent')
            road_id = traci.vehicle.getRoadID('agent')
            lane_id = road_id + '_0'
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            next_tls_info = traci.vehicle.getNextTLS('agent')
            tls_id, tls_index, dist, tls_state = next_tls_info[0]
            if tls_state == "r" or tls_state == "y":
                tls_state = 1
            else:
                tls_state = 0


            
            s_ = np.zeros(s_dim)
            s_[0] = car_spd / 17.67
            s_[1] = (car_a - (-4.5)) / (2.59 - (-4.5))
            s_[2] = SOC_new
            #s_[3] = occupancy
            #s_[4] = tls_state
            ddpg.store_transition(s, action, r, s_)
            # print(total_step)
            if total_step > MEMORY_CAPACITY:
            #if total_step > 1:
                #print("learning")
                ddpg.learn()

            s = s_
            #print("s------------------:",s)
            ep_reward_all += r
            Reward_list_all.append(r)

            total_step += 1
            
            

            SOC = SOC_new
            cost_Engine = -(ep_reward / 0.72 / 1000)
            cost_all = -(ep_reward_all / 0.72 / 1000)
            
        cost_Engine += ((SOC < SOC_origin)*(SOC_origin - SOC)*(201.6 * 6.5)* 3600/ (42600000)/ 0.72)
        #cost_Engine_list.append(cost_Engine)
        #cost_Engine_100Km_list.append(cost_Engine * (100 / (total_milage)))
        writer.add_scalar("Reward", ep_reward_all, i)
        writer.add_scalar("Engine Cost", cost_Engine, i)
        writer.add_scalar("SOC-final(every episode)", SOC, i)
        print("Episode:",i," cost_Engine: %.3f" % cost_Engine," reward: %.3f" % (ep_reward_all)," SOC-final: %.3f" % SOC,)
                
        traci.close()
        #print("---------------------------------------------------------------------------------------------------------------")
    ddpg.savemodel(time_string)
    writer.close()




def test_ddpg():
    log_dir = os.path.join("runs_test", time_string)
    writer = SummaryWriter(log_dir=log_dir)
    s_dim = 3
    a_dim = 1
    a_bound = 1
    ddpg = DDPG(s_dim, a_dim, a_bound, writer)
    ddpg.loadmodel(time_string)


    Prius = Prius_model()
    cost_sum = 0
    #path = "Data_Standard Driving Cycles/Prius_source_data"
    #path_list = os.listdir(path)
    #random_data = np.random.randint(0, len(path_list))
    #base_data = path_list[random_data]
    #data = scio.loadmat(path + "/" + base_data)
    #car_spd_one = data["speed_vector"]

    traci.start(sumoCmd)
    for sim_step in range(601):
        traci.simulationStep()
    SOC = 0.65
    SOC_origin = SOC
    ep_reward = 0
    ep_reward_all = 0
    SOC_data = []
    action_data = []
    P_req_list = []
    P_out_list = []
    Eng_spd_list = []
    Eng_trq_list = []
    Eng_pwr_list = []
    Eng_pwr_opt_list = []
    Gen_spd_list = []
    Gen_trq_list = []
    Gen_pwr_list = []
    Mot_spd_list = []
    Mot_trq_list = []
    Mot_pwr_list = []
    Batt_pwr_list = []
    inf_batt_list = []
    inf_batt_one_list = []
    Reward_list = []
    Reward_list_all = []
    T_list = []
    Mot_eta_list = []
    Gen_eta_list = []
    car_spd_list = []
    car_a_list = []
    car_spd = traci.vehicle.getSpeed('agent')
    car_a = traci.vehicle.getAcceleration('agent')
    road_id = traci.vehicle.getRoadID('agent')
    lane_id = road_id + '_0'
    occupancy = traci.lane.getLastStepOccupancy(lane_id)
    next_tls_info = traci.vehicle.getNextTLS('agent')
    tls_id, tls_index, dist, tls_state = next_tls_info[0]
    if tls_state == "r" or tls_state == "y":
        tls_state = 1
    else:
        tls_state = 0
    s = np.zeros(s_dim)
    s[0] = car_spd / 17.67
    s[1] = (car_a - (-4.5)) / (2.59 - (-4.5))
    s[2] = SOC
    #s[3] = occupancy
    #s[4] = tls_state
    action_noise_type = "None"
    param_noise_scale = None

    while traci.simulation.getTime() < end_time:
        #print(str(j) + "/", car_spd_one.shape[1])
        action, action_no_noise = ddpg.choose_action(
            s, 1, param_noise_scale, False, action_noise_type
        )
        
        a = action
        action_data.append(a)
        Eng_pwr_opt = a * 56000


        out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
        car_spd_list.append(car_spd)
        car_a_list.append(car_a)
        P_req_list.append(float(out["P_req"]))
        P_out_list.append(float(out["P_out"]))
        Eng_spd_list.append(float(out["Eng_spd"]))
        Eng_trq_list.append(float(out["Eng_trq"]))
        Eng_pwr_list.append(float(out["Eng_pwr"]))
        Eng_pwr_opt_list.append(float(out["Eng_pwr_opt"]))
        Mot_spd_list.append(float(out["Mot_spd"]))
        Mot_trq_list.append(float(out["Mot_trq"]))
        Mot_pwr_list.append(float(out["Mot_pwr"]))
        Gen_spd_list.append(float(out["Gen_spd"]))
        Gen_trq_list.append(float(out["Gen_trq"]))
        Gen_pwr_list.append(float(out["Gen_pwr"]))
        Batt_pwr_list.append(float(out["Batt_pwr"]))
        inf_batt_list.append(int(out["inf_batt"]))
        inf_batt_one_list.append(int(out["inf_batt_one"]))
        Mot_eta_list.append(float(out["Mot_eta"]))
        Gen_eta_list.append(float(out["Gen_eta"]))
        T_list.append(float(out["T"]))
        SOC_new = float(out["SOC"])
        SOC_data.append(SOC_new)
        cost = float(cost)
        r = -cost
        ep_reward += r
        Reward_list.append(r)

        if SOC_new < 0.6 :
            r = -((350 * ((0.6 - SOC_new) ** 2)) + cost)
        
        traci.simulationStep()


        car_spd = traci.vehicle.getSpeed('agent')
        car_a = traci.vehicle.getAcceleration('agent')
        road_id = traci.vehicle.getRoadID('agent')
        lane_id = road_id + '_0'
        occupancy = traci.lane.getLastStepOccupancy(lane_id)
        next_tls_info = traci.vehicle.getNextTLS('agent')
        tls_id, tls_index, dist, tls_state = next_tls_info[0]
        if tls_state == "r" or tls_state == "y":
            tls_state = 1
        else:
            tls_state = 0
        
        s_ = np.zeros(s_dim)
        s_[0] = car_spd / 17.67
        s_[1] = (car_a - (-4.5)) / (2.59 - (-4.5))
        s_[2] = SOC_new
        #s_[3] = occupancy
        #s_[4] = tls_state
        s = s_
        ep_reward_all += r
        Reward_list_all.append(r)
        cost_Engine = -(ep_reward / 0.72 / 1000)

        SOC = SOC_new
    
    with pd.ExcelWriter(f"output{time_string}.xlsx") as data_writer:
        pd.DataFrame(car_spd_list).to_excel(data_writer, sheet_name="car_spd_list", index=False)
        pd.DataFrame(car_a_list).to_excel(data_writer, sheet_name="car_a_list", index=False)
        pd.DataFrame(Eng_pwr_opt_list).to_excel(data_writer, sheet_name="Eng_pwr_opt_list", index=False)
        pd.DataFrame(action_data).to_excel(data_writer, sheet_name="action_data", index=False)
        pd.DataFrame(Reward_list).to_excel(data_writer, sheet_name="Reward_list", index=False)
        pd.DataFrame(Reward_list_all).to_excel(data_writer, sheet_name="Reward_list_all", index=False)
        pd.DataFrame(SOC_data).to_excel(data_writer, sheet_name="SOC_data", index=False)
        pd.DataFrame(P_req_list).to_excel(data_writer, sheet_name="P_req_list", index=False)
        pd.DataFrame(P_out_list).to_excel(data_writer, sheet_name="P_out_list", index=False)
        pd.DataFrame(Eng_spd_list).to_excel(data_writer, sheet_name="Eng_spd_list", index=False)
        pd.DataFrame(Eng_trq_list).to_excel(data_writer, sheet_name="Eng_trq_list", index=False)
        pd.DataFrame(Eng_pwr_list).to_excel(data_writer, sheet_name="Eng_pwr_list", index=False)
        pd.DataFrame(Mot_spd_list).to_excel(data_writer, sheet_name="Mot_spd_list", index=False)
        pd.DataFrame(Mot_trq_list).to_excel(data_writer, sheet_name="Mot_trq_list", index=False)
        pd.DataFrame(Mot_pwr_list).to_excel(data_writer, sheet_name="Mot_pwr_list", index=False)
        pd.DataFrame(Gen_spd_list).to_excel(data_writer, sheet_name="Gen_spd_list", index=False)
        pd.DataFrame(Gen_trq_list).to_excel(data_writer, sheet_name="Gen_trq_list", index=False)
        pd.DataFrame(Gen_pwr_list).to_excel(data_writer, sheet_name="Gen_pwr_list", index=False)
        pd.DataFrame(Batt_pwr_list).to_excel(data_writer, sheet_name="Batt_pwr_list", index=False)
        pd.DataFrame(inf_batt_list).to_excel(data_writer, sheet_name="inf_batt_list", index=False)
        pd.DataFrame(inf_batt_one_list).to_excel(data_writer, sheet_name="inf_batt_one_list", index=False)
        pd.DataFrame(Mot_eta_list).to_excel( data_writer, sheet_name="Mot_eta_list", index=False)
        pd.DataFrame(Gen_eta_list).to_excel(data_writer, sheet_name="Gen_eta_list", index=False)
        pd.DataFrame(T_list).to_excel(data_writer, sheet_name="T_list", index=False) 
    cost_Engine += ((SOC < SOC_origin)*(SOC_origin - SOC)*(201.6 * 6.5)* 3600/ (42600000)/ 0.72)  
    print("  reward:", ep_reward_all, "  Engine_cost:", cost_Engine, "  SOC:", SOC)
    traci.close()
    


time_string=time_string1
time_string="2023-10-02_18-10-49_____1"
print("time_string:",time_string)
#run_ddpg()
print("train is done, test:")
test_ddpg()

'''
time_string=time_string2
print("time_string:",time_string)
run_ddpg()
time_string=time_string3
print("time_string:",time_string)
run_ddpg()
time_string=time_string4
print("time_string:",time_string)
run_ddpg()
time_string=time_string5
print("time_string:",time_string)
run_ddpg()
time_string=time_string6
print("time_string:",time_string)
run_ddpg()
time_string=time_string7
print("time_string:",time_string)
run_ddpg()
time_string=time_string8
print("time_string:",time_string)
run_ddpg()
time_string=time_string9
print("time_string:",time_string)
run_ddpg()
time_string=time_string10
print("time_string:",time_string)
run_ddpg()
'''



print("all is done")