from pretrain.utils import get_model_config
from ..env import ModelEnv

def make_env(args):
    env = ModelEnv(args, get_model_config(args.model_name, args.custom_model_name, args.dataset))
    return env

def make_vec_envs(args, num_agents):
    envs = [make_env(args) for i in range(num_agents)]
    
