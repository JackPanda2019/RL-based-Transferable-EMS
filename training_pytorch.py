import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as scio
from Prius_model_new import Prius_model

device = torch.device("cuda")

MAX_EPISODES = 100
LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64
RENDER = False


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
    def __init__(self, state_dim, action_dim, action_bound):
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

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state)
        action = action.squeeze().cpu().numpy()
        return action

    def learn(self):
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

        a_loss = -self.critic(bs, self.actor(bs)).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

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

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state))
        self.memory.store(transition)

    def savemodel(self):
        torch.save(self.actor.state_dict(), "Checkpoints/source/actor_model.pt")
        torch.save(self.critic.state_dict(), "Checkpoints/source/critic_model.pt")


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.data = [None] * capacity
        self.total_priority = 0

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.e) ** self.a

    def store(self, transition):
        if self.data is None:
            self.data = np.zeros((self.capacity, transition.shape[0]))
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1
        self.tree.add(max_priority, transition)
        self.data[self.tree.data_pointer] = transition
        self.tree.data_pointer += 1
        if self.tree.data_pointer >= self.capacity:
            self.tree.data_pointer = 0

    def sample(self, batch_size):
        batch_idx = np.empty((batch_size,), dtype=np.int32)
        batch_memory = np.empty((batch_size, self.tree.data[0].size))
        ISWeights = np.empty((batch_size, 1))

        self.total_priority = self.tree.total_p()

        priority_segment = self.total_priority / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_priority = (
            np.min(self.tree.tree[-self.tree.capacity :]) + 1e-5
        ) / self.total_priority

        if min_priority == 0:
            min_priority = 1e-5

        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            prob = priority / self.total_priority
            ISWeights[i, 0] = np.power(prob / min_priority, -self.beta)
            batch_idx[i] = index
            batch_memory[i, :] = data

        return batch_idx, batch_memory, ISWeights

    def batch_update(self, tree_idx, abs_td_error):
        abs_td_error += self.e
        clipped_errors = np.minimum(abs_td_error, 1.0)
        priorities = self._get_priority(clipped_errors)

        for ti, p in zip(tree_idx, priorities):
            self.tree.update(ti, p)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.capacity - 1 + self.data_pointer
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        return (
            leaf_index,
            self.tree[leaf_index],
            self.data[parent_index - self.capacity + 1],
        )

    def total_p(self):
        return self.tree[0]


def run_ddpg():
    s_dim = 3
    a_dim = 1
    a_bound = 1
    ddpg = DDPG(s_dim, a_dim, a_bound)
    total_step = 0
    step_episode = 0
    mean_reward_all = 0
    cost_Engine_list = []
    cost_all_list = []
    cost_Engine_100Km_list = []
    mean_reward_list = []
    std_reward_list = []
    list_even = []
    list_odd = []
    mean_discrepancy_list = []
    SOC_final_list = []

    mu1 = 0
    sigma1 = 0.03
    threshold = 0.2
    factor = 1.01
    Prius = Prius_model()
    for i in range(MAX_EPISODES):
        path = "Data_Standard Driving Cycles/Prius_source_data"
        path_list = os.listdir(path)
        random_data = np.random.randint(0, len(path_list))
        base_data = path_list[random_data]
        data = scio.loadmat(path + "/" + base_data)
        car_spd_one = data["speed_vector"]
        total_milage = np.sum(car_spd_one) / 1000

        SOC = 0.65
        SOC_origin = SOC
        ep_reward = 0
        ep_reward_all = 0
        mileage = 0
        step_episode += 1
        SOC_data = []
        P_req_list = []
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
        car_spd = car_spd_one[:, 0]
        car_a = car_spd_one[:, 0] - 0
        s = np.zeros(s_dim)
        s[0] = car_spd / 24.1683
        s[1] = (car_a - (-1.6114)) / (1.3034 - (-1.6114))
        s[2] = SOC

        action_noise_type = "None"
        param_noise_scale = np.random.normal(mu1, sigma1)

        for j in range(car_spd_one.shape[1] - 1):
            print(str(i) + " ---> " + str(j) + "/", car_spd_one.shape[1])
            action = ddpg.choose_action(s)

            if param_noise_scale > 0 and j == 0:
                param_noise = np.random.normal(0, param_noise_scale)
                action = np.clip(action + param_noise, -a_bound, a_bound)

            Eng_pwr_opt = (action.item() + a_bound) * 56000

            out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
            P_req_list.append(float(out["P_req"]))
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

            if SOC_new < 0.6 or SOC_new > 0.85:
                r = -((350 * ((0.6 - SOC_new) ** 2)) + cost)

            car_spd = car_spd_one[:, j + 1]
            car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
            s_ = np.zeros(s_dim)
            s_[0] = car_spd / 24.1683
            s_[1] = (car_a - (-1.6114)) / (1.3034 - (-1.6114))
            s_[2] = SOC_new
            ddpg.store_transition(s, action, r, s_)

            if total_step > MEMORY_CAPACITY:
                ddpg.learn()

            s = s_
            ep_reward_all += r
            Reward_list_all.append(r)

            total_step += 1
            SOC = SOC_new
            cost_Engine = -(ep_reward / 0.72 / 1000)
            cost_all = -(ep_reward_all / 0.72 / 1000)

            if j == (car_spd_one.shape[1] - 2):
                SOC_final_list.append(SOC)
                mean_reward = np.mean(Reward_list_all)
                mean_reward_list.append(mean_reward)
                std_reward = np.std(Reward_list_all, ddof=1)
                std_reward_list.append(std_reward)
                cost_Engine += (
                    (SOC < SOC_origin)
                    * (SOC_origin - SOC)
                    * (201.6 * 6.5)
                    * 3600
                    / (42600000)
                    / 0.72
                )
                cost_Engine_list.append(cost_Engine)
                cost_Engine_100Km_list.append(cost_Engine * (100 / (total_milage)))
                cost_all += (
                    (SOC < SOC_origin)
                    * (SOC_origin - SOC)
                    * (201.6 * 6.5)
                    * 3600
                    / (42600000)
                    / 0.72
                )
                cost_all_list.append(cost_all)
                print(
                    "Episode:",
                    i,
                    " cost_Engine: %.3f" % cost_Engine,
                    " reward: %.3f" % -(ep_reward_all / 100),
                    " SOC-final: %.3f" % SOC,
                )

        ddpg.savemodel()

    SOC_final_arr = np.array(SOC_final_list)
    np.savetxt("./soc.txt", SOC_final_arr)
    cost_Engine_arr = np.array(cost_Engine_list)
    np.savetxt("./cost_Engine.txt", cost_Engine_arr)
    cost_all_arr = np.array(cost_all_list)
    np.savetxt("./cost_all.txt", cost_all_arr)
    mean_reward_arr = np.array(mean_reward_list)
    np.savetxt("./mean_reward.txt", mean_reward_arr)
    std_reward_arr = np.array(std_reward_list)
    np.savetxt("./std_reward.txt", std_reward_arr)


run_ddpg()
