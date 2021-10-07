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
    
    # set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    envs = make_vec_envs(args.game, args.seed, args.num_processes,
                         args.gamma, args.log_dir, torch_device, False)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(torch_device)

    agent = A2C_Agent(
         actor_critic,
         args.value_loss_coef,
         args.entropy_coef,
         lr=args.lr,
         eps=args.eps,
         alpha=args.alpha,
         max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                             envs.observation_space.shape, envs.action_space,
                             actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(torch_device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.model)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.game + ".pt"))
        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalzie(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.game, args.seed,
                    args.num_processes, eval_log_dir, torch_device)

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
