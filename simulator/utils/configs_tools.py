"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
from uu import Error


def get_defaults_yaml_args(MARL_algo, predictor_algo, env):
    """Load config file for user-specified algo and env.
    Args:
        MARL_algo: (str) MARL Algorithm name.
        predictor_algo: (str) Predictor name.
        env: (str) Environment name.
    Returns:
        MARL_algo_args: (dict) Algorithm config.
        predictor_algo_args: (dict) Predictor config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    if MARL_algo is None:
        MARL_algo_args = {}
    else:
        MARL_algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{MARL_algo}.yaml")
        with open(MARL_algo_cfg_path, "r", encoding="utf-8") as file:
            MARL_algo_args = yaml.load(file, Loader=yaml.FullLoader)
    if predictor_algo is None:
        predictor_algo_args = {}
    else:
        predict_algo_cfg_path = os.path.join(base_path, "configs", "preds_cfgs", f"{predictor_algo}.yaml")
        with open(predict_algo_cfg_path, "r", encoding="utf-8") as file:
            predictor_algo_args = yaml.load(file, Loader=yaml.FullLoader)
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return MARL_algo_args, predictor_algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "bottleneck":
        task = f"{env_args['scenario']}"
    elif env == "bottleneck_attack":
        task = f"{env_args['scenario']}"
    elif env == "bottleneck_new":
        task = f"{env_args['scenario']}"
    return task


def init_dir(env, env_args, mode, algo, exp_name, seed, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        logger_path,
        env,
        task,
        mode,
        algo,
        exp_name,
        "-".join(["seed-{:0>5}".format(seed), hms_time]),
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir, config_type):
    """Save the configuration of the program."""
    config = {"main_args": args, f"{config_type}_algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, f"{config_type}_config.json"), "w", encoding="utf-8") as out:
        out.write(output)
