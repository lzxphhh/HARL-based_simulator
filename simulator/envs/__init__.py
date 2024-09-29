from absl import flags
from simulator.envs.bottleneck.bottleneck_logger_marl import BottleneckLogger
from simulator.envs.bottleneck.bottleneck_logger_predictor import BottleneckLoggerPre
from simulator.envs.bottleneck_new.bottleneck_logger_marl import BottleneckNewLogger
from simulator.envs.bottleneck_new.bottleneck_logger_predictor import BottleneckNewLoggerPre
from simulator.envs.bottleneck_attack.bottleneck_attack_logger_marl import BottleneckAttackLogger
from simulator.envs.bottleneck_attack.bottleneck_attack_logger_predictor import BottleneckAttackLoggerPre

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "bottleneck_marl": BottleneckLogger,
    "bottleneck_predictor": BottleneckLoggerPre,
    "bottleneck_new_marl": BottleneckNewLogger,
    "bottleneck_new_predictor": BottleneckNewLoggerPre,
    "bottleneck_attack_marl": BottleneckAttackLogger,
    "bottleneck_attack_predictor": BottleneckAttackLoggerPre,
}
