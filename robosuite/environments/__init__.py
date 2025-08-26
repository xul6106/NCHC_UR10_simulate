from .base import REGISTERED_ENVS, MujocoEnv

ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
from robosuite.environments.manipulation.ur10_task import DualTableTask

# 在這裡將您的環境類別添加到 REGISTERED_ENVS 字典中
REGISTERED_ENVS['DualTableTask'] = DualTableTask