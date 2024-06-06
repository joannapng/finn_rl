import os
import math
import torch
import random
import argparse
from copy import deepcopy
from agent.ddpg import DDPG
from train.env import ModelEnv
from finn.util.basic import part_map
from pretrain.utils import get_model_config

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152']

parser = argparse.ArgumentParser(description = 'Train RL Agent')

### ----- TARGET MODEL ------ ###
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model')

### ----- DATASET PARAMETERS ----- ###
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing')
parser.add_argument('--num-workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning')

### ----- FINETUNER PARAMETERS ----- ###
parser.add_argument('--finetuning-epochs', default = 5, type = int, help = 'Finetuning epochs')
parser.add_argument('--print-every', default = 100, type = int, help = 'How frequent to print progress')
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training learning rate')
parser.add_argument('--weight-decay', default = 0, type = float, help = 'Weight decay for optimizer')
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

### ----- QUANTIZATION PARAMETERS ----- ###
parser.add_argument('--scale-factor-type', default='float_scale', choices=['float_scale', 'po2_scale'], help = 'Type for scale factors (default: float)')
parser.add_argument('--act-bit-width', default=4, type=int, help = 'Default activations bit width (default: 4)')
parser.add_argument('--weight-bit-width', default=4, type=int, help = 'Default weight bit width (default: 4)')
parser.add_argument('--bias-bit-width', default=8, choices=[32, 16, 8], help = 'Bias bit width (default: 8)')
parser.add_argument('--bias-corr', default=True, action = 'store_true', help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
parser.add_argument('--hidden1', default = 300, type = int, help = 'Hidden num of first fully connected layer')
parser.add_argument('--hidden2', default = 300, type = int, help = 'Hidden num of second fully connected layer')
parser.add_argument('--lr_c', default = 1e-3, type = float, help = 'Learning rate for actor')
parser.add_argument('--lr_a', default = 1e-4, type = float, help = 'Learning rate for critic')
parser.add_argument('--warmup', default = 20, type = float, help = 'Time without training but only filling the replay memory')
parser.add_argument('--discount', default=1., type=float, help='')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
parser.add_argument('--init_delta', default=0.5, type=float, help='initial variance of truncated normal distribution')
parser.add_argument('--delta_decay', default=0.99, type=float, help='delta decay during exploration')
parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
parser.add_argument('--output', default='./logs', type=str, help='')

parser.add_argument('--num-episodes', default = 100, type = int, help = 'Number of episodes (passes over the entire network) to train the agent for')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce')
parser.add_argument('--init_w', default=0.003, type=float, help='')
parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')

### --- DESIGN --- ###
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--freq', type = float, default = 200.0, help = 'Frequency in MHz')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps when target is accuracy')

def train(num_episode, agent, env, output, debug=False):
    # best record
    best_reward = -math.inf
    best_policy = []

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory

    text_writer.write('# episode,reward,acc,fps,avg utilization,policy\n')
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(num_episode / 10) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
        

        if done:  # end of episode
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            text_writer.write('{},{},{},{},{},{}\n'.format(episode, episode_reward, info['accuracy'], info['fps'], info['avg_util'], info['strategy']))
            
            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy
        
            text_writer.flush()

    text_writer.close()
    agent.eval()
    agent.save(output)
    return best_policy, best_reward

args = parser.parse_args()
args.fpga_part = part_map[args.board]

text_writer = open(os.path.join(args.output, 'log.csv'), 'w')
print('==> Output path: {}...'.format(args.output))

# set seed to reproduce
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'GPU' and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# create environment
env = ModelEnv(args, get_model_config(args.model_name, args.dataset))

nb_actions = env.action_space.shape[-1]
nb_states = env.observation_space.shape[-1]

# create agent
agent = DDPG(nb_states, nb_actions, args)

# train agent
best_policy, best_reward = train(args.num_episodes, agent, env, args.output)
print('best_reward: ', best_reward)
print('best_policy: ', best_policy)
