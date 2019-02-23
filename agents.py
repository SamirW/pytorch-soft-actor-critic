import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy


class SACAgent(object):

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, agent_id, 
                policy_type='Gaussian', 
                hidden_size=256, 
                alpha=0.1, 
                lr=0.0003,
                automatic_entropy_tuning=False):

        # Store agent parameters
        self.agent_id = agent_id
        self.policy_type = policy_type
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Q network
        self.critic = QNetwork(num_in_critic, hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Policy network and critic
        if self.policy_type == "Gaussian":
            self.alpha = alpha
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given
            # in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = - \
                    torch.prod(torch.Tensor(action_space.shape)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            else:
                pass

            self.policy = GaussianPolicy(
                num_in_pol, num_out_pol, hidden_size)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

            self.value = ValueNetwork(num_in_pol, hidden_size)
            self.value_target = ValueNetwork(num_in_pol, hidden_size)
            self.value_optim = Adam(self.value.parameters(), lr=lr)
            hard_update(self.value_target, self.value)
        else:
            self.policy = DeterministicPolicy(
                num_in_pol, num_out_pol, hidden_size)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

            self.critic_target = QNetwork(num_in_critic, hidden_size)
            hard_update(self.critic_target, self.critic)

    def reset(self):
        self.policy.randomize()
        self.critic.randomize()
        if self.policy_type == "Gaussian":
            self.value.randomize()
            hard_update(self.value_target, self.value)
        else:
            hard_update(self.critic_target, self.critic)

    def select_action(self, state, eval=False):
        if eval == False:
            self.policy.train()
            action, _, _, _, _ = self.policy.sample(state)
        else:
            self.policy.eval()
            _, _, _, action, _ = self.policy.sample(state)
            if self.policy_type == "Gaussian":
                action = torch.tanh(action)
            else:
                pass
        return action[0]

    def get_params(self):
        params = dict()

        params['policy'] = self.policy.state_dict()
        params['critic'] = self.critic.state_dict()
        params['policy_optim'] = self.policy_optim.state_dict()
        params['critic_optim'] = self.critic_optim.state_dict()

        if self.policy_type == "Gaussian":
            params['policy_type'] = "Gaussian"
            params['value'] = self.value.state_dict()
            params['value_optim'] = self.value_optim.state_dict()
            params['target_value'] = self.target_value.state_dict()
        else:
            params['policy_type'] = "Deterministic"
            params['target_critic'] = self.target_critic.state_dict()

    def load_params(self, params):
        self.policy_type = params['policy_type']

        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.policy_optim.load_state_dict(params['policy_optim'])
        self.critic_optim.load_state_dict(params['critic_optim'])

        if self.policy_type == "Gaussian":
            self.value.load_state_dict(params['value'])
            self.value_optim.load_state_dict(params['value_optim']) 
            self.target_value.load_state_dict(params['target_value'])
        else:
            self.target_critic.load_state_dict(params['target_critic'])