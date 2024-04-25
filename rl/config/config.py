from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class Config:
    LR: float = 0.001
    BATCH_SIZE: int = 1024
    EPOCHS: int = 10
    NUM_WORKERS: int = 4
    NUM_ENVS: int = 128
    NUM_STEPS: int = 128 
    TOTAL_TIMESTEPS: float = 1e7
    FC_DIM_SIZE: int = 128
    GRU_HIDDEN_DIM: int = 128
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    SCALE_CLIP_EPS: bool = False
    ENT_COEF: float = 0.0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.25
    ACTIVATION: str = 'relu'
    OBS_WITH_AGENT_ID: bool = True
    MAP_NAME: str = '2s3z'
    SEED: int = 0
    # ENV_KWARGS: dict = {} 
    ANNEAL_LR: bool = False
    overwrite: bool = False
    ckpt_freq: int = 50
    render_freq: int = 50

    # WandB Params
    WANDB_MODE: str = 'run'
    ENTITY: str = ""
    PROJECT: str = 'waymax_saphne'

    # DO NOT CHANGE THIS. It is dark magic?
    max_num_objects: int = 32


    # DO NOT CHANGE THESE. They will be set automatically in the code.
    NUM_ACTORS: int = -1
    MINIBATCH_SIZE: int = -1
    NUM_UPDATES: int = -1
    exp_dir: str = ""
    ckpt_dir: str = ""
    vid_dir: str = ""

    
@dataclass
class EnjoyConfig(Config):
    random_agent: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="enjoy", node=EnjoyConfig)
