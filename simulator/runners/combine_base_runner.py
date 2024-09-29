"""Base runner for on-policy algorithms."""

import time
import os
import numpy as np
import torch
import setproctitle
from simulator.common.valuenorm import ValueNorm
from simulator.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from simulator.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from simulator.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from simulator.common.buffers.on_policy_local_prediction_buffer import OnPolicyLocalPredictionBuffer
from simulator.common.buffers.on_policy_global_prediction_buffer import OnPolicyGlobalPredictionBuffer
from simulator.algorithms.actors import ALGO_REGISTRY
from simulator.algorithms.predictors.local_prediction import LOC_PREDICTION
from simulator.algorithms.predictors.global_prediction import GLO_PREDICTION
from simulator.algorithms.critics.v_critic import VCritic
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


class CombineBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, MARL_algo_args, predictor_algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.MARL_algo_args = MARL_algo_args
        self.predictor_algo_args = predictor_algo_args
        self.env_args = env_args

        # 读取MARL算法相关config
        self.hidden_sizes = MARL_algo_args["model"]["hidden_sizes"]  # MLP隐藏层神经元数量
        self.rnn_hidden_size = self.hidden_sizes[-1]  # RNN隐藏层神经元数量
        self.recurrent_n = MARL_algo_args["model"]["recurrent_n"]  # RNN的层数
        self.action_aggregation = MARL_algo_args["algo"]["action_aggregation"]  # 多维动作空间的聚合方式，如mean/prod
        self.share_param = MARL_algo_args["algo"]["share_param"]  # actor是否共享参数
        self.fixed_order = MARL_algo_args["algo"]["fixed_order"]  # 是否固定agent的策略更新顺序
        set_seed(MARL_algo_args["seed"])  # 设置随机种子
        self.device = init_device(MARL_algo_args["device"])  # 设置设备

        # 读取predictor算法相关config
        self.pre_hidden_sizes = predictor_algo_args["model"]["hidden_sizes"]  # MLP隐藏层神经元数量
        self.pre_rnn_hidden_size = self.pre_hidden_sizes[-1]  # RNN隐藏层神经元数量
        self.pre_recurrent_n = predictor_algo_args["model"]["recurrent_n"]  # RNN的层数
        self.pre_share_param = predictor_algo_args["algo"]["share_param"]  # local predictors是否共享参数
        self.pre_fixed_order = predictor_algo_args["algo"]["fixed_order"]  # 是否固定agent的策略更新顺序

        # train, not render 说明在训练，不在eval
        if not self.MARL_algo_args["render"]["use_render"]:
            # 初始化运行路径，日志路径，保存路径，tensorboard路径
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["MARL_algo"],
                args["predictor_algo"],
                args["exp_name"],
                MARL_algo_args["seed"]["seed"],
                logger_path=MARL_algo_args["logger"]["log_dir"],
            )
            # 保存algo，env args，algo args所有config
            save_config(args, MARL_algo_args, predictor_algo_args, env_args, self.run_dir)

        # set the title of the process
        # 设置进程的标题
        setproctitle.setproctitle(
            str(args["mode"]) + "-" + str(args["MARL_algo"]) + "-" + str(args["predictor_algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # 使用env tools中的函数创建训练/测试/render环境 （调取环境+插入env config）
        if self.MARL_algo_args["render"]["use_render"]:
            # 创建单线程render环境
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], MARL_algo_args["seed"]["seed"], env_args)
        else:
            # 创建多线程训练环境
            self.envs = make_train_env(
                args["env"],
                MARL_algo_args["seed"]["seed"],
                MARL_algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            # 创建多线程测试环境
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    MARL_algo_args["seed"]["seed"],
                    MARL_algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if MARL_algo_args["eval"]["use_eval"]
                else None
            )
        # 默认使用EP作为state_type
        # EP：EnvironmentProvided global state (EP)：环境提供的全局状态
        # FP：Featured-Pruned Agent-Specific Global State (FP)： 特征裁剪的特定智能体全局状态(不同agent的全局状态不同, 需要agent number)
        self.state_type = env_args.get("state_type", "EP")
        # TODO： EP or FP need to be added to customized env

        # 智能体数量
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        print("num_agents: ", self.num_agents)

        # actor和local predictor相关
        self.actor = []
        self.local_predictor = []
        # actor共享参数
        if self.share_param:
            # 初始化actor网络，进入mappo.py
            agent = ALGO_REGISTRY[args["algo"]](
                {**MARL_algo_args["model"], **MARL_algo_args["algo"]},  # yaml里model和algo的config打包作为args进入OnPolicyBase
                self.envs.observation_space[0], # 单个agent的观测空间
                self.envs.action_space[0], # 单个agent的动作空间
                device=self.device,
            )
            # 因为共享参数，所以self.actor列表中只有一个actor，即所有agent共用一套actor网络
            self.actor.append(agent)

            # 初始化local_predictor网络，进入local_prediction.py
            agent_prediction = LOC_PREDICTION(
                {**predictor_algo_args["model"], **predictor_algo_args["algo"]},  # yaml里model和algo的config打包作为args
                self.envs.observation_space[0],  # 单个agent的观测空间
                self.envs.action_space[0],  # 单个agent的动作空间
                device=self.device,
            )
            self.local_predictor.append(agent_prediction)

            # 因为共享参数，agent之间的观测空间和动作空间都要同构
            for agent_id in range(1, self.num_agents):
                # 所以self.envs.observation_space作为a list of obs space for each agent应该保持一致
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                # 所以self.envs.action_space list of act space for each agent应该保持一致
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
                # self.actor是一个list，里面有N个一模一样的actor
                self.local_predictor.append(self.local_predictor[0])
                # self.local_predictor是一个list，里面有N个一模一样的predictor

        # actor不共享参数
        else:
            for agent_id in range(self.num_agents):
                # 给每一个agent初始化actor网络，进入mappo.py 【根据其不同的obs_dim和act_dim】
                agent = ALGO_REGISTRY[args["algo"]](
                    {**MARL_algo_args["model"], **MARL_algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                # 因为不共享参数，所以self.actor列表中有N个actor，所有agent每人一套actor网络
                self.actor.append(agent)
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
        if self.MARL_algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            # 给每一个agent创立buffer，初始化buffer，进入OnPolicyActorBuffer
            self.local_predictor_buffer = []
            # 给每一个local predictor创立buffer，初始化buffer，进入OnPolicyLocalPredictionBuffer
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    # yaml里model和algo的config打包作为args进入OnPolicyActorBuffer
                    {**MARL_algo_args["train"], **MARL_algo_args["model"]},
                    # 【根据其不同的obs_dim和act_dim】
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                # self.actor_buffer列表中有N个buffer，所有agent每人一套buffer
                self.actor_buffer.append(ac_bu)
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

            # 创建centralized critic网络
            self.critic = VCritic(
                # yaml里model和algo的config打包作为args进入VCritic
                {**MARL_algo_args["model"], **MARL_algo_args["algo"]},
                # 中心式的值函数centralized critic的输入是单个agent拿到的share_observation_space dim
                share_observation_space,
                device=self.device,
            )

            # 创建centralized critic网络的buffer（1个）
            # MAPPO trick: 原始论文section 5.2
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                # EP的全局状态是所有agent的状态的拼接，所以所有agent的share_observation_space dim是一样的
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**MARL_algo_args["train"], **MARL_algo_args["model"], **MARL_algo_args["algo"]},
                    share_observation_space,
                )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.

                # FP的全局状态是EP+IND的prune版本（包含全局状态，agent specific状态，并且删除了冗余状态），因此每个agent不一样 #TODO：还没看
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**MARL_algo_args["train"], **MARL_algo_args["model"], **MARL_algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                # TODO： 或许EP+IND的非prune版本？
                raise NotImplementedError

            # 创建global predictor网络的buffer（1个）
            self.global_predictor_buffer = OnPolicyGlobalPredictionBuffer(
                {**predictor_algo_args["train"], **predictor_algo_args["model"], **predictor_algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )

            # MAPPO trick: 原始论文 section 5.1 - PopArt？
            if self.MARL_algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            # 环境的logger
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, MARL_algo_args, predictor_algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # 可以restore之前训练到一半的模型继续训练
        if self.MARL_algo_args != {}:
            if self.MARL_algo_args["train"]["model_dir"] is not None:  # restore model
                self.restore_MARL_model()
        if self.predictor_algo_args != {}:
            if self.predictor_algo_args["train"]["model_dir"] is not None:
                self.restore_predictor_model()


    def restore_MARL_model(self):
        """Restore model parameters."""
        if self.share_param:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(self.MARL_algo_args["train"]["model_dir"])
                    + "/actor_agent"
                    + '0'
                    + ".pt"
                )
                self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        else:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(self.MARL_algo_args["train"]["model_dir"])
                    + "/actor_agent"
                    + str(agent_id)
                    + ".pt"
                )
                self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not self.MARL_algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(self.MARL_algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
            )
            self.critic.critic.load_state_dict(policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.MARL_algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def restore_predictor_model(self):
        for agent_id in range(self.num_agents):
            policy_local_predictor_state_dict = torch.load(
                str(self.predictor_algo_args["train"]["model_dir"])
                + "/local_predictor"
                + str(agent_id)
                + ".pt"
            )
            self.local_predictor[agent_id].local_predictor.load_state_dict(policy_local_predictor_state_dict)
        policy_global_predictor_state_dict = torch.load(
            str(self.predictor_algo_args["train"]["model_dir"]) + "/global_predictor" + ".pt"
        )
        self.global_predictor.global_predictor.load_state_dict(policy_global_predictor_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.MARL_algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.MARL_algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()