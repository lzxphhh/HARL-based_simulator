"""Train an algorithm."""
import argparse
import json
from simulator.utils.configs_tools import get_defaults_yaml_args, update_args

"""
# More Run/Debug --> Modify Run Configuration --> Script Parameters
--MARL_algo <MARL_ALGO> 
--predictor_algo <Predict_ALGO>
--env <ENV> 
--exp_name <EXPERIMENT NAME>
--load_MARL_config <TUNED CONFIG PATH>
--load_predictor_config <TUNED CONFIG PATH>

# train MARL with trained predictor
--MARL_algo mappo --env bottleneck_attack --exp_name 0830_mappo_improve --load_predictor_config ./config.json
--MARL_algo happo --env bottleneck --exp_name 0830_happo_improve --load_predictor_config ./config.json
or
# train predictor with trained MARL
--predictor_algo GAT --env bottleneck --exp_name 0830_prediction --load_MARL_config ./config.json
or
# test MARL & predictor
--load_MARL_config ./config.json --load_predictor_config ./config.json
"""
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 运行模式
    parser.add_argument(
        "--mode",
        type=str,
        default="train_MARL",
        choices=[
            "train_MARL",
            "train_predictor",
            "test",
        ],
        help="Run mode. Choose from: train_MARL, train_predictor, test.",
    )
    # 使用什么MARL算法
    parser.add_argument(
        "--MARL_algo",
        type=str,
        default="",
        choices=[
            "happo",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, mappo.",
    )
    # 使用什么预测器
    parser.add_argument(
        "--predictor_algo",
        type=str,
        default="",
        choices=[
            "GAT",
        ],
        help="If set, train the predictor. Choose from: GAT.",
    )
    # 使用什么环境
    parser.add_argument(
        "--env",
        type=str,
        default="bottleneck",
        choices=[
            "bottleneck",
            "bottleneck_attack",
            "bottleneck_new"
        ],
        help="Environment name. Choose from: bottleneck, bottleneck-attack, bottleneck_new.",
    )
    # 实验名称
    parser.add_argument(
        "--exp_name", type=str, default="bottleneck", help="Experiment name."
    )
    # 是否使用MARL config file
    parser.add_argument(
        "--load_MARL_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    # 是否使用预测器config file
    parser.add_argument(
        "--load_predictor_config",
        type=str,
        default="",
        help="If set, load existing experiment config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    # 读取命令行参数
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    def load_config_file(filepath):
        with open(filepath, encoding="utf-8") as file:
            return json.load(file)

    def get_config_args(config, args, config_type):
        args[f"{config_type}_algo"] = config["main_args"][f"{config_type}_algo"]
        algo_args = config[f"{config_type}_algo_args"]
        if args["mode"] == "test" and config_type == "MARL":
            args["env"] = config["main_args"]["env"]
        env_args = config.get("env_args", {})
        return algo_args, env_args

    MARL_algo_args, predictor_algo_args, env_args = {}, {}, {}

    # Load MARL config if provided
    if args["load_MARL_config"]:
        MARL_config = load_config_file(args["load_MARL_config"])
        MARL_algo_args, env_args = get_config_args(MARL_config, args, "MARL")
        args["MARL_algo"] = MARL_config["main_args"]["MARL_algo"]

    # Load Predictor config if provided
    if args["load_predictor_config"]:
        predictor_config = load_config_file(args["load_predictor_config"])
        predictor_algo_args, _ = get_config_args(predictor_config, args, "predictor")
        args["predictor_algo"] = predictor_config["main_args"]["predictor_algo"]

    # If only one config is provided, load default values for the missing one
    if args["mode"] == "train_MARL":
        MARL_algo_args, _, env_args = get_defaults_yaml_args(args["MARL_algo"], None, args["env"])
    if args["mode"] == "train_predictor":
        _, predictor_algo_args, env_args = get_defaults_yaml_args(None, args["predictor_algo"], args["env"])

    # If no configs are provided, load default values from yaml
    if not args["load_MARL_config"] and not args["load_predictor_config"]:
        MARL_algo_args, predictor_algo_args, env_args = get_defaults_yaml_args(args["predictor_algo"],
                                                                               args["MARL_algo"], args["env"])

    update_args(unparsed_dict, MARL_algo_args, predictor_algo_args, env_args)  # update args from command line

    # start training
    from simulator.runners import MARL_RUNNER_REGISTRY
    # , PREDICTION_RUNNER_REGISTRY)
    # from simulator.runners.on_policy_combine_runner import OnPolicyCombineRunner

    runner = MARL_RUNNER_REGISTRY[args["MARL_algo"]](args, MARL_algo_args, predictor_algo_args, env_args)

    # if args["MARL_algo"] != "":
    #     runner = MARL_RUNNER_REGISTRY[args["MARL_algo"]](args, MARL_algo_args, predictor_algo_args, env_args)
    # elif args["predictor_algo"] != "":
    #     runner = PREDICTION_RUNNER_REGISTRY[args["predictor_algo"]](args, MARL_algo_args, predictor_algo_args, env_args)
    # else:
    #     runner = OnPolicyCombineRunner(args, MARL_algo_args, predictor_algo_args, env_args)

    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
