"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from simulator.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    # 调取原始环境
    def get_env_fn(rank):
        def init_env():
            if env_name == "bottleneck":
                from simulator.envs.bottleneck.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)

            elif env_name == "bottleneck_attack":
                from simulator.envs.bottleneck_attack.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)
            elif env_name == "bottleneck_new":
                from simulator.envs.bottleneck_new.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    # 生成多线程环境
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "bottleneck":
                from simulator.envs.bottleneck.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)

            elif env_name == "bottleneck_attack":
                from simulator.envs.bottleneck_attack.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)
            elif env_name == "bottleneck_new":
                from simulator.envs.bottleneck_new.bottleneck_env import BOTTLENECKEnv

                env = BOTTLENECKEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "bottleneck":
        from simulator.envs.bottleneck.bottleneck_env import BOTTLENECKEnv

        env = BOTTLENECKEnv(env_args)
        manual_render = False
        env.seed(seed * 60000)
    elif env_name == "bottleneck_attack":
        from simulator.envs.bottleneck_attack.bottleneck_env import BOTTLENECKEnv

        env = BOTTLENECKEnv(env_args)
        manual_render = False
        env.seed(seed * 60000)
    elif env_name == "bottleneck_new":
        from simulator.envs.bottleneck_new.bottleneck_env import BOTTLENECKEnv

        env = BOTTLENECKEnv(env_args)
        manual_render = False
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "bottleneck":
        return envs.n_agents
    elif env == "bottleneck_attack":
        return envs.n_agents
    elif env == "bottleneck_new":
        return envs.n_agents
