from simulator.common.base_logger import BaseLogger
import time
import numpy as np
import csv
import os


# TODO： 统计以下指标：
# 1. 平均速度
# 2. 每30个episode的done的原因
# 3. 每30个episode的平均长度

class BottleneckAttackLoggerPre(BaseLogger):

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(BottleneckAttackLoggerPre, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.train_episode_rewards = None
        self.train_episode_local_prediction_errors = None
        self.train_episode_global_prediction_errors = None
        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_rewards = None
        self.done_episode_lens = None
        self.done_episodes_local_prediction_errors = None
        self.done_episodes_global_prediction_errors = None

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
        self.train_episode_local_prediction_errors = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_global_prediction_errors = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_local_prediction_errors = np.zeros(self.n_rollout_threads)
        self.done_episodes_global_prediction_errors = np.zeros(self.n_rollout_threads)
        save_csv_dir = self.run_dir + '/csv'
        self.csv_path = save_csv_dir + '/episode_info.csv'
        self.prediction_csv_path = save_csv_dir + '/prediction_info.csv'
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
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            action_losss,
        ) = data
        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)
        # 并行环境中的每个环境的step reward （n_env_threads, ）
        reward_env = np.mean(rewards, axis=1).flatten()
        # 并行环境中的每个环境的episode reward （n_env_threads, ）累积
        self.train_episode_rewards += reward_env
        # 并行环境中的每个环境的episode len （n_env_threads, ）累积
        self.one_episode_len += 1

        for t in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_rewards[t] = self.train_episode_rewards[t]
                self.train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos[t] = infos[t][0]

                # 存一下这个已经done的episode的episode长度
                self.done_episode_lens[t] = self.one_episode_len[t]
                if not self.use_world_model:
                    self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度

                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                # assert round(self.done_episode_infos[t]['episode_return'], 2) == \
                #        round(self.done_episodes_rewards[t], 2) or \
                #        round(self.done_episode_infos[t]['episode_return'],2) == \
                #        round(self.done_episodes_rewards[t],2), 'episode reward not match'
                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[t]['step_time'] == self.done_episode_lens[t]+1, 'episode len not match'

    def per_step_prediction(self, prediction_data):
        """Process data per step."""

        (
            obs,
            share_obs,
            dones,  # (n_threads, n_agents)
            actions,
            local_prediction_errors,
            rnn_states_local_prediction,
            global_prediction_errors,
            rnn_states_global_prediction,
        ) = prediction_data
        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)
        # 并行环境中的每个环境的step prediction error （n_env_threads, ）
        local_prediction_error_env = np.mean(local_prediction_errors, axis=1).flatten()
        global_prediction_error_env = np.mean(global_prediction_errors, axis=1).flatten()
        # 并行环境中的每个环境的episode prediction error （n_env_threads, ）累积
        self.train_episode_local_prediction_errors += local_prediction_error_env
        self.train_episode_global_prediction_errors += global_prediction_error_env
        # 并行环境中的每个环境的episode len （n_env_threads, ）累积

        for t in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_local_prediction_errors[t] = self.train_episode_local_prediction_errors[t]/self.one_episode_len[t] if self.one_episode_len[t] != 0 else 0
                self.train_episode_local_prediction_errors[t] = 0  # 归零这个以及done的episode的local_prediton_error
                self.done_episodes_global_prediction_errors[t] = self.train_episode_global_prediction_errors[t]/self.one_episode_len[t] if self.one_episode_len[t] != 0 else 0
                self.train_episode_global_prediction_errors[t] = 0  # 归零这个以及done的episode的global_prediton_error
                self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度


    def episode_log(
            self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer,
            save_collision, save_episode_step
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
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # 记录每个episode的平均avergae reward 和 average step和collision rate
        collision_count = 0
        done_episode_lens = []
        for each_env_info in self.done_episode_infos:
            if each_env_info['done_reason'] == 'collision':
                collision_count += 1
            else:
                done_episode_lens.append(each_env_info['step_time']-1)
        average_collision_rate = collision_count / len(self.done_episode_infos)
        average_episode_return = np.mean(self.done_episodes_rewards)
        average_episode_step = np.mean(self.done_episode_lens)
        # average_episode_step = np.mean(done_episode_lens) if done_episode_lens else 0

        # self.writter.add_scalars(
        #     "average_collision_rate",
        #     {"average_collision_rate": average_collision_rate},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "average_collision_rate", average_collision_rate, self.total_num_steps
        )
        print(
            "Some episodes done, average collision rate is {}.\n".format(
                average_collision_rate
            )
        )

        # self.writter.add_scalars(
        #     "average_episode_length",
        #     {"average_episode_length": average_episode_step},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "average_episode_length", average_episode_step, self.total_num_steps
        )

        print(
            "Some episodes done, average episode length is {}.\n".format(
                average_episode_step
            )
        )

        print(
            "Some episodes done, average episode reward is {}.\n".format(
                average_episode_return
            )
        )
        # self.writter.add_scalars(
        #     "train_episode_rewards",
        #     {"aver_rewards": average_episode_return},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "train_episode_rewards", average_episode_return, self.total_num_steps
        )

        # 记录每个episode的平均 step reward
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)
        print(
            "Average step reward is {}.\n".format(
                critic_train_info["average_step_rewards"]
            )
        )
        # self.writter.add_scalars(
        #     "average_step_rewards",
        #     {"average_step_rewards": critic_train_info["average_step_rewards"]},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "average_step_rewards", critic_train_info["average_step_rewards"], self.total_num_steps
        )
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.total_num_steps, average_collision_rate, average_episode_step, average_episode_return])

        if average_collision_rate <= save_collision and average_episode_step <= save_episode_step:
            return True, self.total_num_steps
        else:
            return False, self.total_num_steps

    def episode_prediction_log(
            self,
            local_world_model_train_infos,
            global_world_model_train_info,
            local_world_model_buffer,
            global_world_model_buffer,
            save_prediction,
            save_prediction_step
    ):
        """Log prediction information for each episode."""
        average_local_prediction_error = np.mean(self.done_episodes_local_prediction_errors)
        average_global_prediction_error = global_world_model_buffer.get_mean_prediction_errors()

        self.log_world_model(local_world_model_train_infos, global_world_model_train_info)
        print(
            "Average local prediction error is {}.\n".format(
                average_local_prediction_error
            )
        )

        print(
            "Average global prediction error is {}.\n".format(
                average_global_prediction_error
            )
        )

        self.writter.add_scalar(
            "average_global_prediction_error", average_global_prediction_error, self.total_num_steps
        )
        with open(self.prediction_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.total_num_steps, average_local_prediction_error, average_global_prediction_error])

        if average_local_prediction_error <= save_prediction and average_global_prediction_error <= save_prediction and self.total_num_steps >= save_prediction_step:
            return True, self.total_num_steps
        else:
            return False, self.total_num_steps
