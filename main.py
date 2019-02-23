import argparse
import time
import gym
import itertools
import torch
import copy
import numpy as np
from torch import Tensor
from torch.autograd import Variable
from sac import SAC
from tensorboardX import SummaryWriter
from normalized_actions import NormalizedActions
from buffer import ReplayBuffer
from make_env import make_env
from utils import DummyVecEnv


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default:False)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.1)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 1024)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--max_ep_length', type=int, default=25, metavar='N',
                    help='maximum number of steps in episode (default: 50)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_update', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
# New arguments
parser.add_argument('--steps_per_update', type=int, default=100, metavar='N',
                    help='number of steps before updating (default: 100)')
parser.add_argument('--flip_ep', type=int, default=1000000, metavar='N',
                    help='episde after which to use flipped environment (default: 10000000)')
parser.add_argument('--n_rollout_threads', type=int, default=1, metavar='N',
                    help='multithreading')
parser.add_argument('--n_training_threads', type=int, default=6, metavar='N',
                    help='multithreading')
args = parser.parse_args()


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    # Assert checks
    assert n_rollout_threads == 1
    assert discrete_action is False

    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    return DummyVecEnv([get_env_fn(0)])

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(args.n_training_threads)

# Environment
env = make_parallel_env(
    args.env_name, args.n_rollout_threads, args.seed, discrete_action=False)
sac = SAC.init_from_env(env, args)

writer = SummaryWriter()

# Memory
replay_buffer = ReplayBuffer(
    args.replay_size, sac.nagents,
    [obsp.shape[0] for obsp in env.observation_space],
    [acsp.shape[0] for acsp in env.action_space])

# Training Loop
total_rewards = []
test_rewards = []
total_numsteps = 0
updates = 0
flip = False

for i_episode in itertools.count():
    # Flip episodes
    if i_episode == args.flip_ep:
        print("Flipping ...")
        replay_buffer = ReplayBuffer(
            config.buffer_length, sac.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space])
        flip = True

    obs = env.reset(flip=flip)
    episode_reward = 0
    ep_step = 0

    while True:
        # rearrange observation to be per agent
        torch_obs = [
            Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
            for i in range(sac.nagents)]

        # Find action
        if args.start_steps > total_numsteps:
            torch_agent_actions = env.sample_action_spaces()  # Sample action from env
        else:
            torch_agent_actions = sac.step(torch_obs)  # Sample action from policy

        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

        # Take action in the environment
        next_obs, rewards, dones, infos = env.step([copy.deepcopy(agent_actions)])

        if ep_step == args.max_ep_length - 1:
            dones = dones + 1

        if (i_episode % 10) == 0:
            time.sleep(0.01)
            env.render()

        # Add to buffer
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

        if len(replay_buffer) > args.batch_size:
            # Number of steps before updating
            if (total_numsteps % args.steps_per_update) == 0:
                # Number of updates per step in environment
                for i in range(args.updates_per_update):
                    for a_i in range(sac.nagents):
                        # Sample a batch from memory
                        sample = replay_buffer.sample(args.batch_size, norm_rews=True)

                        # Update parameters of all the networks for each agent
                        q_value, value_loss, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                            sac.update_parameters(a_i, sample, updates)

                        writer.add_scalar(
                            'agent_{}/critic/q_value'.format(a_i), q_value, updates)
                        writer.add_scalar(
                            'agent_{}/loss/value'.format(a_i), value_loss, updates)
                        writer.add_scalar(
                            'agent_{}/loss/critic_1'.format(a_i), critic_1_loss, updates)
                        writer.add_scalar(
                            'agent_{}/loss/critic_2'.format(a_i), critic_2_loss, updates)
                        writer.add_scalar(
                            'agent_{}/loss/policy'.format(a_i), policy_loss, updates)
                        writer.add_scalar(
                            'agent_{}/loss/entropy_loss'.format(a_i), ent_loss, updates)
                        writer.add_scalar(
                            'agent_{}/entropy_temprature/alpha'.format(a_i), alpha, updates)
                    updates += 1

        # For next step
        obs = next_obs
        total_numsteps += args.n_rollout_threads
        episode_reward += np.sum(rewards) / sac.nagents
        ep_step += 1

        if dones.all():
            break

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    total_rewards.append(episode_reward)
    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, np.round(total_rewards[-1], 2),
                                                                                   np.round(np.mean(total_rewards[-100:]), 2)))

    if i_episode % 10 == 0 and args.eval == True:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            print("eval")
            action = agent.select_action(state, eval=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
            if done:
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        test_rewards.append(episode_reward)
        print("----------------------------------------")
        print("Test Episode: {}, reward: {}".format(
            i_episode, test_rewards[-1]))
        print("----------------------------------------")

env.close()