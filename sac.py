import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, ValueNetwork


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.num_inputs = num_inputs
        self.action_space = action_space.shape[0]
        self.gamma = args.gamma
        self.tau = args.tau
        self.scale_R = args.scale_R
        self.reparam = args.reparam
        self.deterministic = args.deterministic
        self.value_update = args.value_update

        self.policy = GaussianPolicy(self.num_inputs, self.action_space, args.hidden_size)
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)

        self.critic = QNetwork(self.num_inputs, self.action_space, args.hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=3e-4)

        if self.deterministic == False:
            self.value = ValueNetwork(self.num_inputs, args.hidden_size)
            self.value_target = ValueNetwork(self.num_inputs, args.hidden_size)
            self.value_optim = Adam(self.value.parameters(), lr=3e-4)
            hard_update(self.value_target, self.value)
            self.value_criterion = nn.MSELoss()
        else:
            self.critic_target = QNetwork(self.num_inputs, self.action_space, args.hidden_size)
            hard_update(self.critic_target, self.critic)

        self.soft_q_criterion = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if deterministic == False:
            _, _, x_t, _, _ = self.policy.evaluate(state)
            action = torch.tanh(x_t)
        else:
            _, _, _, x_t, _ = self.policy.evaluate(state)
            action = torch.tanh(x_t)
        action = action.detach().cpu().numpy()
        return action[0]



    def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, updates):
        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        mask_batch = torch.FloatTensor(np.float32(mask_batch))

        expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)

        new_action, log_prob, x_t, mean, log_std = self.policy.evaluate(state_batch, reparam=self.reparam)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        if self.deterministic == False:
            expected_value = self.value(state_batch)
            target_value = self.value_target(next_state_batch)
            next_q_value = self.scale_R * reward_batch + mask_batch * self.gamma * target_value
        else:
            target_critic_1, target_critic_2 = self.critic_target(next_state_batch, new_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            next_q_value = self.scale_R * reward_batch + mask_batch * self.gamma * target_critic

        q1_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach())
        q2_value_loss = self.soft_q_criterion(expected_q2_value, next_q_value.detach())
        q1_new, q2_new = self.critic(state_batch, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)

        if self.deterministic == False:
            next_value = expected_new_q_value - log_prob
            value_loss = self.value_criterion(expected_value, next_value.detach())
            log_prob_target = expected_new_q_value - expected_value
        else:
            log_prob_target = expected_new_q_value

        if self.reparam == True:
            policy_loss = (log_prob - expected_new_q_value).mean()
        else:
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()

        policy_loss += mean_loss + std_loss

        self.critic_optim.zero_grad()
        q1_value_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        q2_value_loss.backward()
        self.critic_optim.step()

        if self.deterministic == False:
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.value_update == 0 and self.deterministic == True:
            soft_update(self.critic_target, self.critic, self.tau)
        elif updates % self.value_update == 0 and self.deterministic == False:
            soft_update(self.value_target, self.value, self.tau)


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))
