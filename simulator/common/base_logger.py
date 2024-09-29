"""Base logger."""

import time
import os
import numpy as np


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        # self.use_world_model = algo_args["train"]["use_world_model"]

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError

    def init(self, episodes):
        """
        Initialize the logger.
        输入：episodes总个数
        """
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []

    def episode_init(self, episode):
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
        ) = data
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
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

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            # self.writter.add_scalars(
            #     "train_episode_rewards",
            #     {"aver_rewards": aver_episode_rewards},
            #     self.total_num_steps,
            # )
            self.writter.add_scalar(
                "train_episode_rewards", aver_episode_rewards, self.total_num_steps
            )
            self.done_episodes_rewards = []

    def eval_init(self):
        """Initialize the logger for evaluation."""

        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                self.one_episode_rewards.append([])
                self.eval_episode_rewards.append([])
        if self.args["mode"] == "train_predictor":
            self.eval_episode_local_prediction_errors = []
            self.one_episode_local_prediction_errors = []
            self.eval_episode_global_prediction_errors = []
            self.one_episode_global_prediction_errors = []
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                self.one_episode_local_prediction_errors.append([])
                self.eval_episode_local_prediction_errors.append([])
                self.one_episode_global_prediction_errors.append([])
                self.eval_episode_global_prediction_errors.append([])

    def eval_MARL_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_predictor_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
            eval_local_prediction_errors,
            eval_global_prediction_errors,
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.one_episode_local_prediction_errors[eval_i].append(eval_local_prediction_errors[eval_i])
            self.one_episode_global_prediction_errors[eval_i].append(eval_global_prediction_errors[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()

    def log_train_MARL(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                # self.writter.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)
                self.writter.add_scalar(agent_k, v, self.total_num_steps)
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            # self.writter.add_scalars(critic_k, {critic_k: v}, self.total_num_steps)
            self.writter.add_scalar(critic_k, v, self.total_num_steps)

    def log_train_predictor(self, local_predictor_train_infos, global_predictor_train_info):
            for agent_id in range(self.num_agents):
                for k, v in local_predictor_train_infos[agent_id].items():
                    agent_k = "local_predictor_%i/" % agent_id + k
                    # self.writter.add_scalars(agent_k, {agent_k: v}, self.total_num_steps)
                    self.writter.add_scalar(agent_k, v, self.total_num_steps)
            for k, v in global_predictor_train_info.items():
                global_k = "global_predictor/" + k
                # self.writter.add_scalars(global_k, {global_k: v}, self.total_num_steps)
                self.writter.add_scalar(global_k, v, self.total_num_steps)

    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                # self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)
                self.writter.add_scalar(k, np.mean(v), self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()