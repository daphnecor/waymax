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
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    NUM_WORKERS: int = 4
    NUM_ENVS: int = 128
    NUM_STEPS: int = 128 
    TOTAL_TIMESTEPS: float = 1e7
    HIDDEN_DIM: int = 128
    HIDDEN_DIM: int = 128
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
    # MAP_NAME: str = '2s3z'
    SEED: int = 0
    ANNEAL_LR: bool = False
    OVERWRITE: bool = False
    CKPT_FREQ: int = 50
    RENDER_FREQ: int = 50
    EXP_NAME: Optional[str] = None

    # ENV_KWARGS: dict = {}  # Use a subconfig for this if we really need it
    OFFROAD: float = -1.0
    OVERLAP: float = 0.0
    LOG_DIVERGENCE: float = 0.0

    # WandB Params
    WANDB_MODE: str = 'run'  # one of: 'offline', 'run', 'dryrun', 'shared', 'disabled', 'online'
    ENTITY: str = ""
    PROJECT: str = 'waymax_saphne'

    # DO NOT CHANGE THIS. It is dark magic?
    MAX_NUM_OBJECTS: int = 8


    # DO NOT CHANGE THESE. They will be set automatically in the code.
    _num_actors: int = -1
    _minibatch_size: int = -1
    _num_updates: int = -1
    _exp_dir: str = ""
    _ckpt_DIR: str = ""
    _vid_dir: str = ""

    
@dataclass
class RenderConfig(Config):
    RANDOM_AGENT: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="render", node=RenderConfig)
