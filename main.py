# main.py
# Codes were based on https://github.com/jmichaux/dqn-pytorch.git

import argparse
from collections import namedtuple

import gym

from wrappers import make_env
from env_utils import make_vec_envs
from deeprl.DQNAgent import DQNAgent
import utils

import torch


def arg_parse():
    desc = "Deep RL Experiment"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='DQN',
                        choices=['DQN', 'A2C'], required=True)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'], required=False)
    parser.add_argument('--game', type=str, default='PongNoFrameskip-v4',
                        choices=['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'],
                        required=False)
    parser.add_argument('--train', type=int, default=20000, required=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-dir', default='/tmp/gym/')
    # Parameters for A2C
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--num-processes', type=int, default=16)
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=None)
    parser.add_argument('--num-env-steps', type=int, default=10e6)
    parser.add_argument('--recurrent-policy', action='store_true', default=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    
    # set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create environment
    if args.game == 'pong':
        env = gym.make("PongNoFrameskip-v4")
    elif args.game == 'breakout':
        env = gym.make("BreakoutNoFrameskip-v4")
    env = make_env(env)

    # create networks
    if args.model == 'DQN':
        agent = DQNAgent(env, args, torch_device)

    if args.mode == 'train':
        agent.train()
    elif args.mode == 'test':
        agent.test()
