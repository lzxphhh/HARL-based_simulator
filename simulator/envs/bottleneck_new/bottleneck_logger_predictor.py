from simulator.common.base_logger import BaseLogger
import time
import numpy as np
import csv
import os


# TODO： 统计以下指标：
# 1. 平均速度
# 2. 每30个episode的done的原因
# 3. 每30个episode的平均长度

class BottleneckNewLoggerPre(BaseLogger):

    def __init__(self, args, predictor_algo_args, env_args, num_agents, writter, run_dir):
        super(BottleneckNewLoggerPre, self).__init__(
            args, predictor_algo_args, env_args, num_agents, writter, run_dir
        )
        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.n_rollout_threads = predictor_algo_args["train"]["n_rollout_threads"]
        self.train_episode_local_prediction_errors = None
        self.train_episode_global_prediction_errors = None
        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_local_prediction_error = None
        self.done_episodes_global_prediction_error = None
        self.done_episode_lens = None

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"

    def init(self, episodes):
        # 初始化logger

        self.start = time.time()
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=int)
        self.time_steps = [1, 2, 3, 4, 5]
        self.categories = ['all', 'CAV', 'HDV']
        self.train_episode_local_prediction_errors = {}
        self.done_episodes_local_prediction_errors = {}
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                self.train_episode_local_prediction_errors[key] = np.zeros(self.algo_args["train"]["n_rollout_threads"])
                self.done_episodes_local_prediction_errors[key] = np.zeros(self.n_rollout_threads)

        self.train_episode_global_prediction_errors = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_global_prediction_error = np.zeros(self.n_rollout_threads)
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

    def per_step(self, data):
        """Process data per step."""

        (
            obs,
            share_obs,
            dones,
            infos,
            values,
            global_prediction_errors,
            actions,
            rnn_states_actor,
            localPre_errs,
            rnn_states_local,  # rnn_states_critic是critic的rnn的hidden state
            rnn_states_global,
        ) = data
        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)

        local_prediction_error = {}
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                local_prediction_error[key] = np.nanmean(localPre_errs[key], axis=1).flatten() if not np.all(np.isnan(localPre_errs[key])) else 0
                self.train_episode_local_prediction_errors[key] += local_prediction_error[key]

        global_prediction_error_env = np.mean(global_prediction_errors, axis=1).flatten()
        self.train_episode_global_prediction_errors += global_prediction_error_env
        # 并行环境中的每个环境的episode len （n_env_threads, ）累积
        self.one_episode_len += 1

        for env in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[env]:
                # 已经done的episode的总reward
                for t in self.time_steps:
                    for category in self.categories:
                        key = f'{t}s_{category}'
                        self.done_episodes_local_prediction_errors[key][env] = self.train_episode_local_prediction_errors[key][env]/(self.one_episode_len[env]-5)
                        self.train_episode_local_prediction_errors[key][env] = 0

                self.done_episodes_global_prediction_error[env] = self.train_episode_global_prediction_errors[env]/self.one_episode_len[env]
                self.train_episode_global_prediction_errors[env] = 0  # 归零这个以及done的episode的global prediction error

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos[env] = infos[env][0]

                # 存一下这个已经done的episode的episode长度
                self.done_episode_lens[env] = self.one_episode_len[env]
                self.one_episode_len[env] = 0  # 归零这个以及done的episode的episode长度

                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[env]['step_time'] == self.done_episode_lens[env]+1, 'episode len not match'

    def episode_log(
            self,
            local_predictor_train_infos,
            global_predictor_train_info,
            local_predictor_buffer,
            global_predictor_buffer,
            save_prediction_error,
            save_prediction_step
    ):
        """Log information for each episode."""
        # 记录训练结束时间
        self.end = time.time()

        # 当前跑了多少time steps
        self.total_num_steps = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()

        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["predictor_algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        """Log prediction information for each episode."""
        self.log_values = {}
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                self.log_values[key] = np.mean(self.done_episodes_local_prediction_errors[key])
                print(
                    f"Average {t}s-{category} local prediction error is {self.log_values[key]}.\n"
                )

        average_global_prediction_error = global_predictor_buffer.get_mean_prediction_errors()

        self.log_train_predictor(local_predictor_train_infos, global_predictor_train_info)

        print(
            "Average global prediction error is {}.\n".format(
                average_global_prediction_error
            )
        )

        self.writter.add_scalar(
            "average_global_prediction_error", average_global_prediction_error, self.total_num_steps
        )
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            write_data = [self.total_num_steps]

            for t in self.time_steps:
                for category in self.categories:
                    key = f'{t}s_{category}'
                    write_data.append(self.log_values[key])
            write_data.append(average_global_prediction_error)
            writer.writerow(write_data)

        if self.log_values['1s_all'] <= save_prediction_error and average_global_prediction_error <= save_prediction_error and self.total_num_steps >= save_prediction_step:
            return True, self.total_num_steps
        else:
            return False, self.total_num_steps
