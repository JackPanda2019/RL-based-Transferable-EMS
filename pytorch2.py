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

device = torch.device("cuda")
now = datetime.datetime.now()
time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", time_string)
writer = SummaryWriter(log_dir=log_dir)

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
        return torch.normal(mean=mu, std=sigma)

    def OrnsteinUhlenbeckActionNoise(self, mu, sigma, theta=0.15, dt=1e-2):
        ou_noise = (
            self.ou_noise_prev
            + theta * (mu - self.ou_noise_prev) * dt
            + sigma * torch.sqrt(torch.tensor(dt)) * torch.normal(mean=mu, std=sigma)
        )
        self.ou_noise_prev = ou_noise
        return ou_noise

    def choose_action(
        self, state, loop, param_noise_scale, param_noise=True, action_noise_type=None
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
            output_noise = output + self.NormalActionNoise(0, 0.05)  # Gaussian noise
        elif action_noise_type == "ou":
            output_noise = output + self.OrnsteinUhlenbeckActionNoise(
                0, 0.15, 0.2, 0.01
            )  # OU noise
        elif action_noise_type == "None":
            output_noise = output

        return output_noise, output_no_noise

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

    def savemodel(self):
        torch.save(self.actor.state_dict(), "Checkpoints/source/actor_model.pt")
        torch.save(self.critic.state_dict(), "Checkpoints/source/critic_model.pt")


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
    s_dim = 3
    a_dim = 1
    a_bound = 1
    ddpg = DDPG(s_dim, a_dim, a_bound, writer)
    total_step = 0
    step_episode = 0
    cost_Engine_list = []
    cost_all_list = []
    cost_Engine_100Km_list = []
    mean_reward_list = []
    std_reward_list = []
    SOC_final_list = []

    mu1 = 0
    sigma1 = 0.06
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
            action, action_no_noise = ddpg.choose_action(
                s, j, param_noise_scale, False, action_noise_type
            )
            adaptive_policy_distance = np.sqrt(
                np.mean(np.square(action - action_no_noise))
            )
            if adaptive_policy_distance > threshold:
                param_noise_scale = param_noise_scale / factor
            else:
                param_noise_scale = param_noise_scale * factor

            a = np.clip(action, 0, 1)

            Eng_pwr_opt = a * 56000

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
            # print(total_step)
            if total_step > MEMORY_CAPACITY:
                # if total_step > 0:
                # print("learn start")
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
                writer.add_scalar("Reward", ep_reward_all, i)
                writer.add_scalar("Engine Cost", cost_Engine, i)
                writer.add_scalar("cost_all", cost_all, i)
                print(
                    "Episode:",
                    i,
                    " cost_Engine: %.3f" % cost_Engine,
                    " reward: %.3f" % -(ep_reward_all / 100),
                    " SOC-final: %.3f" % SOC,
                )

        ddpg.savemodel()
    writer.close()
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
