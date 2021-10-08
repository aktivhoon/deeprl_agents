# main.py
# Codes were based on https://github.com/jmichaux/dqn-pytorch.git

import argparse
from collections import namedtuple, deque
import time

import gym

#from wrappers import make_env
from env_utils import make_vec_envs
from deeprl.DQNAgent import DQNAgent
from deeprl.A2C.net import Policy
from deeprl.A2C.agent import A2C_Agent
from deeprl.A2C.storage import RolloutStorage

import utils

import torch
import numpy as np
import os


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
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False)
    parser.add_argument('--save-dir', default='./trained_models/')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = make_vec_envs(args.game, args.seed, args.num_processes,
                         args.gamma, args.log_dir, torch_device, False)
    
    # Actor Critic Neural Net
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(torch_device)

    # Agent which learns the game
    agent = A2C_Agent(
         envs,
         actor_critic,
         args.value_loss_coef,
         args.entropy_coef,
         args,
         lr=args.lr,
         eps=args.eps,
         alpha=args.alpha,
         max_grad_norm=args.max_grad_norm)
    
    agent.train()
    """
    Temporarily neglect original code we made for DQN

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
    """
