# Base Agent

import numpy as np
import copy
from collections import namedtuple
from itertools import count
import math
import os
import time
import random

import gym

from .memory import ReplayMemory

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class BaseAgent:
    def __init__(self, env, args, torch_device):
        # environment setting
        self.env = env
        self.num_actions = env.action_space.n
        
        self.device = torch_device

        # hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.final_epsilon = 0.05
        self.exploration_steps = 1000000
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.lr = 1e-4
        self.target_update = 1000
        self.initial_replay = 10000
        self.memory_size = 10000
        self.train_episodes = args.train
        self.train_interval = 4

        self.memory = ReplayMemory(self.memory_size)

        self.save_path = './results/' + args.model + '/' + args.game + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.total_reward = 0.0
        self.total_loss = 0.0

        self.steps_done = 0
        self.writer = SummaryWriter(log_dir = self.save_path)

    def select_action(self):
        """
            The agent selects the action using this function.
        """
        return


    def inner_loop(self):
        """
            Inner loop for optimization
        """
        return


    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, render = False):
        """
            Trains the Q-network
        """
        return

    
    def test(self, render = True):
        """
            Test the Q-network
        """
        return