import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time
import os
from collections import deque

import utils
from deeprl.A2C.storage import RolloutStorage

class A2C_Agent():
    def __init__(self,
                 envs,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 args,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None):

        self.envs = envs
        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)

        self.rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                      envs.observation_space.shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

        self.num_env_steps = args.num_env_steps
        self.num_steps = args.num_steps
        self.num_processes = args.num_processes
        self.gamma = args.gamma
        self.use_proper_time_limits = args.use_proper_time_limits

        self.save_interval = args.save_interval
        self.save_dir = args.save_dir

        self.log_dir = os.path.expanduser(args.log_dir)
        self.eval_log_dir = self.log_dir + "_eval"
        utils.cleanup_log_dir(self.log_dir)
        utils.cleanup_log_dir(self.eval_log_dir)
        self.log_interval = args.log_interval
        self.eval_interval = args.eval_interval
        self.game = args.game

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def train(self):
        # Initialize all environments
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(
            self.num_env_steps) // self.num_steps // self.num_processes

        for j in range(num_updates):
            # Do for rollout step size
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                obs, reward, done, infos = self.envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()
            
            self.rollouts.compute_returns(next_value, self.gamma, self.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.update(self.rollouts)

            self.rollouts.after_update()

            if (j % self.save_interval == 0
                    or j == num_updates - 1) and self.save_dir != "":
                save_path = os.path.join(self.save_dir, 'a2c')
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                   self.actor_critic,
                   getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                ], os.path.join(save_path, self.game + ".pt"))

            if j % self.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

            if (self.eval_interval is not None and len(episode_rewards) > 1
                    and j % self.eval_interval == 0):
                obs_rms = utils.get_vec_normalzie(self.envs).obs_rms
                evaluate(actor_critic, obs_rms, self.game, self.seed,
                        self.num_processes, self.eval_log_dir, torch_device)
