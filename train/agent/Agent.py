from env.ModelEnv import ModelEnv
from env.utils import make_vec_envs

class Agent(object):
    def __init__(self, args):
        envs = make_vec_envs(args, args.num_agents)
