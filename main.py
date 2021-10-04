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
    parser.add_argument('--game', type=str, default='pong',
                        choices=['pong', 'breakout'], required=False)
    parser.add_argument('--train', type=int, default=20000, required=False)
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
