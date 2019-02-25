import numpy as np
import itertools
import argparse
import torch
import copy
import time
import gym
import os
from torch import Tensor
from pathlib import Path
from algorithms.sac import SAC
from utils.logging import set_log
from utils.misc import DummyVecEnv
from utils.make_env import make_env
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from utils.normalized_actions import NormalizedActions


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('env_name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('model_name', default="eval_graph",
                    help='model of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
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
parser.add_argument('--num_eps', type=int, default=1e5, metavar='N',
                    help='maximum number of episodes (default: 1e5)')
parser.add_argument('--max_ep_length', type=int, default=25, metavar='N',
                    help='maximum number of steps in episode (default: 50)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_update', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_eps', type=int, default=500, metavar='N',
                    help='Epss sampling random actions (default: 500)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
# New arguments
parser.add_argument('--steps_per_update', type=int, default=100, metavar='N',
                    help='number of steps before updating (default: 100)')
parser.add_argument('--flip_ep', type=int, default=1000000, metavar='N',
                    help='episode after which to use flipped environment (default: 10000000)')
parser.add_argument('--n_rollout_threads', type=int, default=1, metavar='N',
                    help='multithreading')
parser.add_argument('--n_training_threads', type=int, default=6, metavar='N',
                    help='multithreading')
# Distill args
parser.add_argument('--distill_ep', type=int, default=1e7, metavar='N',
                    help='episode at which to distill agents (default: 1500)')
parser.add_argument('--distill_num', type=int, default=1024, metavar='N',
                    help='number of times to perform distillation (default: 1024)')
parser.add_argument('--distill_batch_size', type=int, default=1024, metavar='N',
                    help='distillation batch size (default: 1024)')
parser.add_argument('--distill_pass_actor', action='store_true', default=False,
                    help='skip actor distillation (default: False)')
parser.add_argument('--distill_pass_critic', action='store_true', default=False,
                    help='skip critic distillation (default: False)')
parser.add_argument('--save_buffer', action='store_true', default=False,
                    help='save replay buffer (default: False)')
parser.add_argument('--log_comment', type=str, default='',
                    help='comment for log file')
args = parser.parse_args()


def make_parallel_env(env_name, n_rollout_threads, seed, discrete_action):
    # Assert checks
    assert n_rollout_threads == 1
    assert discrete_action is False

    def get_env_fn(rank):
        def init_env():
            env = make_env(env_name, discrete_action=discrete_action)
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

# Saving
model_dir = Path('./models') / args.env_name / args.model_name
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)

# Log files
if args.log_comment == '':
    args.log_comment = curr_run

args.log_name = \
    "env::%s_seed::%s_comment::%s_log" % (
        args.env_name, str(args.seed), args.log_comment)

run_dir = model_dir / curr_run
log_dir = run_dir / 'logs'
os.makedirs(str(log_dir))
log = set_log(args, model_dir)
writer = SummaryWriter(str(log_dir))

# Start run
for i_episode in itertools.count():
    # Flip episodes
    if i_episode == args.flip_ep:
        print("Flipping ...")
        replay_buffer = ReplayBuffer(
            args.replay_size, sac.nagents,
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
        if args.start_eps > i_episode:
            torch_agent_actions = env.sample_action_spaces()  # Sample action from env
        else:
            torch_agent_actions = sac.step(
                torch_obs)  # Sample action from policy

        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

        # Take action in the environment
        next_obs, rewards, dones, infos = env.step(
            [copy.deepcopy(agent_actions)])

        if ep_step == args.max_ep_length - 1:
            dones = dones + 1

        # Add to buffer
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

        if len(replay_buffer) > args.batch_size:
            # Number of steps before updating
            if (total_numsteps % args.steps_per_update) == 0:
                # Number of updates per step in environment
                for i in range(args.updates_per_update):
                    for a_i in range(sac.nagents):
                        # Sample a batch from memory
                        sample = replay_buffer.sample(
                            args.batch_size, norm_rews=True)

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
        episode_reward += np.sum(rewards) / sac.nagents / args.max_ep_length
        ep_step += 1

        if dones.all():
            break

    if i_episode > args.num_eps:
        break

    total_rewards.append(episode_reward)

    writer.add_scalar('reward/train', episode_reward, i_episode)
    log[args.log_name].info(
        " Train Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, np.round(total_rewards[-1], 2),
                                                                                        np.round(np.mean(total_rewards[-100:]), 2)))
    if i_episode % 10 == 0 and args.eval == True:
        obs = env.reset(flip=flip)
        test_ep_step = 0
        episode_reward = 0
        while True:
            # Render
            if i_episode % 100 == 0:
                env.render()

            # Find action
            torch_obs = [
                Variable(torch.Tensor(
                    np.vstack(obs[:, i])), requires_grad=False)
                for i in range(sac.nagents)]
            torch_agent_actions = sac.step(torch_obs, eval=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # Take action
            next_obs, rewards, dones, infos = env.step(
                [copy.deepcopy(agent_actions)])

            # Next step and bookeeping
            obs = next_obs
            test_ep_step += 1
            episode_reward += np.sum(rewards) / \
                sac.nagents / args.max_ep_length

            if test_ep_step == args.max_ep_length - 1:
                break

        test_rewards.append(episode_reward)

        writer.add_scalar('reward/test', episode_reward, i_episode)
        print("----------------------------------------")
        log[args.log_name].info(
            "Test Episode: {}, reward: {}".format(i_episode, test_rewards[-1]))
        print("----------------------------------------")

    if (i_episode + 1) == args.distill_ep:
        print("************Distilling***********")

        # Save buffer
        os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
        sac.save(str(run_dir / 'incremental' / ('before_distillation.pt')))

        # Distill
        sac.distill(args.distill_num, args.distill_batch_size, replay_buffer,
                    pass_actor=args.distill_pass_actor, pass_critic=args.distill_pass_critic)

if args.save_buffer:
    print("*******Saving Replay Buffer******")
    import pickle
    with open(str(run_dir / 'replay_buffer.pkl'), 'wb') as output:
        pickle.dump(replay_buffer, output, -1)

print("*******Saving and Closing*******")
sac.save(str(run_dir / 'model.pt'))
env.close()
writer.export_scalars_to_json(str(log_dir / 'summary.json'))
writer.close()
