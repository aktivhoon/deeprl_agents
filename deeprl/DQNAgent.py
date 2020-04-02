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
from .BaseAgent import BaseAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, in_channels = 4, n_actions = 14):
        """
            Initialize Deep Q Network
            
            Arguments:
                in_channels (int): number of input channels
                n_actions (init): number of outputs
        """

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DQNAgent(BaseAgent):
    def __init__(self, env, args, torch_device):
        super(DQNAgent, self).__init__(env, args, torch_device)

        # create networks
        self.policy_net = DQN(n_actions = self.num_actions).to(self.device)
        self.target_net = DQN(n_actions = self.num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = self.lr)

    def select_action(self, state):
        self.steps_done += 1
        sample = random.random()
        if self.epsilon >= sample or self.steps_done < self.initial_replay:
            action = torch.tensor([[random.randrange(4)]], device = self.device, dtype = torch.long)
        else:
            action = self.policy_net(state.to('cuda')).max(1)[1].view(1, 1)

        if self.epsilon > self.final_epsilon and self.steps_done >= self.initial_replay:
            self.epsilon -= self.epsilon_step
        
        return action

    def inner_loop(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device = 'cuda'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device = 'cuda'), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device = self.device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None]).to('cuda')
        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.total_loss += loss
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, render = False):
        for episode in range(self.train_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            self.total_reward = 0.0
            self.total_loss = 0.0
            for t in count():
                action = self.select_action(state)

                if render:
                    self.env.render()

                obs, reward, done, info = self.env.step(action)

                self.total_reward += reward

                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device = self.device)

                self.memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state

                if self.steps_done > self.initial_replay:
                    # Train Network
                    if self.steps_done % self.train_interval == 0:
                        self.inner_loop()

                    if self.steps_done % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    break
            self.writer.add_scalar('Loss', self.total_loss, episode)
            self.writer.add_scalar('Reward', self.total_reward, episode)
            if episode % 500 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward:{}'.format(self.steps_done, episode, t, self.total_reward))
    
        self.env.close()
        self.writer.close()
        return
    
    def test(self, render = True):
        self.env = gym.wrappers.Monitor(self.env, self.save_path + 'videos/' + 'final_video', video_callable = lambda episode_id: True, force = True)
        for episode in range(1):
            obs = self.env.reset()
            state = self.get_state(obs)
            self.total_reward = 0.0

            for t in count():
                action = self.policy_net(state.to('cuda')).max(1)[1].view(1, 1)

                if render:
                    self.env.render()
                    time.sleep(0.02)

                obs, reward, done, info = self.env.step(action)

                self.total_reward += reward

                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None

                state = next_state

                if done:
                    print("Finished Episode {} with reward {}".format(episode, total_reward))
                    break

        self.env.close()
        return