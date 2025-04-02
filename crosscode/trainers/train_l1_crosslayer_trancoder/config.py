from crosscode.trainers.config_common import BaseExperimentConfig, BaseSparseCoder
from crosscode.trainers.train_l1_crosscoder.config import L1TrainConfig


class L1CrossLayerTranscoderConfig(BaseSparseCoder):
    dec_init_norm: float = 0.1


class L1CrossLayerTranscoderExperimentConfig(BaseExperimentConfig):
    transcoder: L1CrossLayerTranscoderConfig
    train: L1TrainConfig
    hookpoints: list[str]
