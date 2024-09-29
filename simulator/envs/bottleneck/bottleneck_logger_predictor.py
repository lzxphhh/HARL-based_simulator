from simulator.common.base_logger import BaseLogger
import time
import numpy as np
import csv
import os


# TODO： 统计以下指标：
# 1. 平均速度
# 2. 每30个episode的done的原因
# 3. 每30个episode的平均长度

class BottleneckLoggerPre(BaseLogger):

    def __init__(self, args, predictor_algo_args, env_args, num_agents, writter, run_dir):
        super(BottleneckLoggerPre, self).__init__(
            args, predictor_algo_args, env_args, num_agents, writter, run_dir
        )
        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.n_rollout_threads = predictor_algo_args["train"]["n_rollout_threads"]
        self.train_episode_rewards = None
        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_rewards = None
        self.done_episode_lens = None

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"

    def init(self, episodes):
        # 初始化logger

        self.start = time.time()
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=int)
        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = np.zeros(self.n_rollout_threads)
        self.done_episode_lens = np.zeros(self.n_rollout_threads)
        self.done_episode_infos = [{} for _ in range(self.n_rollout_threads)]
        save_csv_dir = self.run_dir + '/csv'
        self.csv_path = save_csv_dir + '/episode_info.csv'
        os.makedirs(save_csv_dir)

        pass

    def episode_init(self, episode):
        # 每个episode开始的时候更新logger里面的episode index

        """Initialize the logger for each episode."""
        self.episode = episode

