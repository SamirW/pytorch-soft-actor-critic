import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy
from agents import SACAgent


class SAC(object):
    def __init__(self, agent_init_params, num_agents,
                 gamma=0.99, tau=0.005, target_update_interval=1,
                 policy_type='Gaussian', hidden_size=256, alpha=0.1, 
                 lr=0.0003, automatic_entropy_tuning=False):

        self.nagents = num_agents
        self.agents = [SACAgent(automatic_entropy_tuning=automatic_entropy_tuning,
                            policy_type=policy_type, hidden_size=hidden_size,
                            alpha=alpha, lr=lr, **params)
                        for params in agent_init_params]
        self.distilled_agent = SACAgent(automatic_entropy_tuning=automatic_entropy_tuning,
                            policy_type=policy_type, hidden_size=hidden_size,
                            alpha=alpha, lr=lr, **agent_init_params[0])
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        
    def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, updates):
        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mask_batch = torch.FloatTensor(np.float32(mask_batch)).unsqueeze(1)

        """
        Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)
        new_action, log_prob, _, mean, log_std = self.policy.sample(state_batch)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
                alpha_logs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.)
                alpha_logs = self.alpha # For TensorboardX logs


            """
            Including a separate function approximator for the soft value can stabilize training.
            """
            expected_value = self.value(state_batch)
            target_value = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (target_value).detach()
        else:
            """
            There is no need in principle to include a separate function approximator for the state value.
            We use a target critic network for deterministic policy and eradicate the value value network completely.
            """
            alpha_loss = torch.tensor(0.)
            alpha_logs = self.alpha  # For TensorboardX logs
            next_state_action, _, _, _, _, = self.policy.sample(next_state_batch)
            target_critic_1, target_critic_2 = self.critic_target(next_state_batch, next_state_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            next_q_value = reward_batch + mask_batch * self.gamma * (target_critic).detach()
        
        
        """
        Soft Q-function parameters can be trained to minimize the soft Bellman residual
        JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        ∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
        """
        q1_value_loss = F.mse_loss(expected_q1_value, next_q_value)
        q2_value_loss = F.mse_loss(expected_q2_value, next_q_value)
        q1_new, q2_new = self.critic(state_batch, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)

        if self.policy_type == "Gaussian":
            """
            Including a separate function approximator for the soft value can stabilize training and is convenient to 
            train simultaneously with the other networks
            Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
            JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - α * log π(at|st)]))^2]
            ∇JV = ∇V(st)(V(st) - Q(st,at) + (α * logπ(at|st)))
            """
            next_value = expected_new_q_value - (self.alpha * log_prob)
            value_loss = F.mse_loss(expected_value, next_value.detach())
        else:
            pass

        """
        Reparameterization trick is used to get a low variance estimator
        f(εt;st) = action sampled from the policy
        εt is an input noise vector, sampled from some fixed distribution
        Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        ∇Jπ = ∇log π + ([∇at (α * logπ(at|st)) − ∇at Q(st,at)])∇f(εt;st)
        """
        policy_loss = ((self.alpha * log_prob) - expected_new_q_value).mean()

        # Regularization Loss
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()

        policy_loss += mean_loss + std_loss

        self.critic_optim.zero_grad()
        q1_value_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        q2_value_loss.backward()
        self.critic_optim.step()

        if self.policy_type == "Gaussian":
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            value_loss = torch.tensor(0.)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        
        """
        We update the target weights to match the current value function weights periodically
        Update target parameter after every n(args.target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
            soft_update(self.critic_target, self.critic, self.tau)

        elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
            soft_update(self.value_target, self.value, self.tau)
        return value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_logs

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, args):

        agent_init_params = []
        agent_id = 0
        for acsp, obsp in zip(env.action_space, env.observation_space):
            # Policy
            num_in_pol = obsp.shape[0]
            num_out_pol = acsp.shape[0]

            # Qnetwork and Value network
            num_in_critic = 0
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env.action_space:
                num_in_critic += oacsp.shape[0]

            agent_init_params.append({'agent_id': agent_id,
                                      'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
            agent_id += 1

        init_dict = {'gamma': args.gamma, 'tau': args.tau, 'lr': args.lr, 'num_agents': agent_id,
                     'policy_type': args.policy, 'hidden_size': args.hidden_size,
                     'alpha': args.alpha, 'agent_init_params': agent_init_params,
                     'automatic_entropy_tuning': args.automatic_entropy_tuning,
                     'target_update_interval': args.target_update_interval}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance