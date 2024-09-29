"""Base runner for prediction."""

import time
import os
import numpy as np
import torch
import setproctitle
from simulator.common.valuenorm import ValueNorm
from simulator.common.buffers.on_policy_local_prediction_buffer import OnPolicyLocalPredictionBuffer
from simulator.common.buffers.on_policy_global_prediction_buffer import OnPolicyGlobalPredictionBuffer
from simulator.algorithms.predictors import Prediction_REGISTRY
from simulator.algorithms.predictors.local_prediction import LOC_PREDICTION
from simulator.algorithms.predictors.global_prediction import GLO_PREDICTION
from simulator.utils.trans_tools import _t2n
from simulator.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from simulator.utils.models_tools import init_device
from simulator.utils.configs_tools import init_dir, save_config
from simulator.envs import LOGGER_REGISTRY


class PredictBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, predictor_algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = predictor_algo_args
        self.env_args = env_args

        # 读取predictor算法相关config
        self.pre_hidden_sizes = predictor_algo_args["model"]["hidden_sizes"]  # MLP隐藏层神经元数量
        self.pre_rnn_hidden_size = self.pre_hidden_sizes[-1]  # RNN隐藏层神经元数量
        self.pre_recurrent_n = predictor_algo_args["model"]["recurrent_n"]  # RNN的层数
        self.pre_share_param = predictor_algo_args["algo"]["share_param"]  # local predictors是否共享参数
        self.pre_fixed_order = predictor_algo_args["algo"]["fixed_order"]  # 是否固定agent的策略更新顺序
        set_seed(predictor_algo_args["seed"])  # 设置随机种子
        self.device = init_device(predictor_algo_args["device"])  # 设置设备

        # train, not render 说明在训练，不在eval
        if not self.algo_args["render"]["use_render"]:
            # 初始化运行路径，日志路径，保存路径，tensorboard路径
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["predictor_algo"],
                args["exp_name"],
                predictor_algo_args["seed"]["seed"],
                logger_path=predictor_algo_args["logger"]["log_dir"],
            )
            # 保存algo，env args，algo args所有config
            save_config(args, predictor_algo_args, env_args, self.run_dir)

        # set the title of the process
        # 设置进程的标题
        setproctitle.setproctitle(
            str(args["predictor_algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # 使用env tools中的函数创建训练/测试/render环境 （调取环境+插入env config）
        if self.algo_args["render"]["use_render"]:
            # 创建单线程render环境
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], predictor_algo_args["seed"]["seed"], env_args)
        else:
            # 创建多线程训练环境
            self.envs = make_train_env(
                args["env"],
                predictor_algo_args["seed"]["seed"],
                predictor_algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            # 创建多线程测试环境
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    predictor_algo_args["seed"]["seed"],
                    predictor_algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if predictor_algo_args["eval"]["use_eval"]
                else None
            )
        # 智能体数量
        self.state_type = env_args.get("state_type", "EP")
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        print("num_agents: ", self.num_agents)

        # local predictor相关
        # local predictor共享参数
        if self.pre_share_param:
            self.local_predictor = []
            # 初始化local_predictor网络，进入local_prediction.py
            agent_prediction = LOC_PREDICTION(
                {**predictor_algo_args["model"], **predictor_algo_args["algo"]},  # yaml里model和algo的config打包作为args
                self.envs.observation_space[0],  # 单个agent的观测空间
                self.envs.action_space[0],  # 单个agent的动作空间
                device=self.device,
            )
            self.local_predictor.append(agent_prediction)
            for agent_id in range(1, self.num_agents):
                # 因为共享参数，agent之间的观测空间和动作空间都要同构
                # 所以self.envs.observation_space作为a list of obs space for each agent应该保持一致
                assert (
                        self.envs.observation_space[agent_id]
                        == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                # 所以self.envs.action_space list of act space for each agent应该保持一致
                assert (
                        self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.local_predictor.append(self.local_predictor[0])
                # self.local_predictor是一个list，里面有N个一模一样的predictor，

        # actor不共享参数
        else:
            self.local_predictor = []
            for agent_id in range(self.num_agents):
                # 给每一个agent初始化local_predictor网络，进入local_prediction.py 【根据其不同的obs_dim和act_dim】
                agent_prediction = LOC_PREDICTION(
                {**predictor_algo_args["model"], **predictor_algo_args["algo"]},  # yaml里model和algo的config打包作为args
                self.envs.observation_space[0],  # 单个agent的观测空间
                self.envs.action_space[0],  # 单个agent的动作空间
                device=self.device,
            )
                # 因为不共享参数，所以self.local_predictor列表中有N个predictor，所有agent每人一套local_predictor网络
                self.local_predictor.append(agent_prediction)
        # global predictor相关
        # 初始化global_predictor网络，进入global_prediction.py
        self.global_predictor = GLO_PREDICTION(
            # yaml里model和algo的config打包作为args进入global_prediction
            {**predictor_algo_args["model"], **predictor_algo_args["algo"]},
            self.envs.share_observation_space[0],
            self.envs.action_space[0],
            device=self.device,
        )
        # 训练
        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.local_predictor_buffer = []
            # 给每一个local predictor创立buffer，初始化buffer，进入OnPolicyLocalPredictionBuffer
            for agent_id in range(self.num_agents):
                lp_bu = OnPolicyLocalPredictionBuffer(
                    # yaml里model和algo的config打包作为args进入OnPolicyLocalPredictionBuffer
                    {**predictor_algo_args["train"], **predictor_algo_args["model"]},
                    # 【根据其不同的obs_dim和act_dim】
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                # self.local_predictor_buffer，所有agent每人一套buffer
                self.local_predictor_buffer.append(lp_bu)

            # 单个agent的share obs space eg: Box(-inf, inf, (54,), float32)
            share_observation_space = self.envs.share_observation_space[0]
            # 创建global predictor网络的buffer（1个）
            self.global_predictor_buffer = OnPolicyGlobalPredictionBuffer(
                {**predictor_algo_args["train"], **predictor_algo_args["model"], **predictor_algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            # 环境的logger
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, predictor_algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # 可以restore之前训练到一半的模型继续训练
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()
