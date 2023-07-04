import gym
import numpy as np
import os
import scipy.io as scio
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import DDPG
import scipy.io as scio
from Prius_model_new import Prius_model


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.current_step = 0
        self.s_dim = 3
        self.a_dim = 1
        self.a_bound = 1
        self.total_step = 0
        self.step_episode = 0
        self.SOC_final_list = []
        self.Prius = Prius_model()
        self.path = "Data_Standard Driving Cycles/Prius_source_data"
        self.path_list = os.listdir(self.path)
        self.mu1 = 0
        self.sigma1 = 0.03

        self.action_space = gym.spaces.Box(
            low=-self.a_bound, high=self.a_bound, shape=(self.a_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.s_dim,), dtype=np.float32
        )

    def reset(self):
        random_data = np.random.randint(0, len(self.path_list))
        base_data = self.path_list[random_data]
        data = scio.loadmat(os.path.join(self.path, base_data))
        car_spd_one = data["speed_vector"]
        self.total_milage = np.sum(car_spd_one) / 1000

        self.SOC = 0.65
        self.SOC_origin = self.SOC
        self.ep_reward = 0
        self.ep_reward_all = 0
        self.mileage = 0
        self.step_episode += 1
        self.SOC_data = []
        self.car_spd = car_spd_one[:, 0]
        self.car_a = car_spd_one[:, 0] - 0
        self.state = np.zeros(self.s_dim)
        self.state[0] = self.car_spd / 24.1683
        self.state[1] = (self.car_a - (-1.6114)) / (1.3034 - (-1.6114))
        self.state[2] = self.SOC

        return self.state

    def step(self, action):
        param_noise_scale = np.random.normal(self.mu1, self.sigma1)
        random_data = np.random.randint(0, len(self.path_list))
        base_data = self.path_list[random_data]
        data = scio.loadmat(os.path.join(self.path, base_data))
        car_spd_one = data["speed_vector"]
        j = self.current_step % car_spd_one.shape[1]

        if param_noise_scale > 0 and self.total_step == 0:
            param_noise = np.random.normal(0, param_noise_scale)
            action = np.clip(action + param_noise, -self.a_bound, self.a_bound)

        Eng_pwr_opt = (action.item() + self.a_bound) * 56000

        out, cost, I = self.Prius.run(self.car_spd, self.car_a, Eng_pwr_opt, self.SOC)
        SOC_new = float(out["SOC"])
        cost = float(cost)
        r = -cost
        self.ep_reward += r
        self.car_spd = car_spd_one[:, j]
        self.car_a = car_spd_one[:, j] - car_spd_one[:, j - 1]
        self.state = np.zeros(self.s_dim)
        self.state[0] = self.car_spd / 24.1683
        self.state[1] = (self.car_a - (-1.6114)) / (1.3034 - (-1.6114))
        self.state[2] = SOC_new
        self.total_step += 1

        if self.SOC_new < 0.6 or self.SOC_new > 0.85:
            r = -((350 * ((0.6 - SOC_new) ** 2)) + cost)

        done = self.total_step == (car_spd_one.shape[1] - 2)
        info = {}

        return self.state, r, done, info

    def render(self):
        # 可选：渲染环境（例如，显示图像）
        pass


def run_ddpg():
    MAX_EPISODES = 100
    LR_A = 0.001
    MEMORY_CAPACITY = 50000
    BATCH_SIZE = 64
    GAMMA = 0.9
    TAU = 0.01

    set_random_seed(0)

    env = DummyVecEnv([lambda: CustomEnv()])

    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LR_A,
        buffer_size=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
    )

    model.learn(total_timesteps=MAX_EPISODES)
    model.save("ddpg_model")


run_ddpg()
