# main.py
# Codes were based on https://github.com/jmichaux/dqn-pytorch.git

import argparse
import numpy as np
import copy
from collections import namedtuple
from itertools import count
import math
import os
import time
import random

import gym

from wrappers import *
from deeprl.DQNAgent import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))

def arg_parse():
    desc = "Deep RL Experiment"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='DQN',
                        choices=['DQN'], required=True)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'], required=False)
    parser.add_argument('--game', type=str, default='pong',
                        choices=['pong', 'breakout'], required=False)
    parser.add_argument('--train', type=int, default=200000, required=False)
    parser.add_argument('--n_dist', type=int, default=31, required=False)
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
        agent.test()
    elif args.mode == 'test':
        agent.test()