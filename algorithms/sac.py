import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update
from utils.model import GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy
from utils.agents import SACAgent

from torch.distributions.kl import kl_divergence


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

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def critics(self):
        return [a.critic for a in self.agents]

    @property
    def values(self):
        return [a.value for a in self.agents]

    def step(self, observations, eval=False):
        return [a.select_action(obs, eval=eval) for a, obs
                in zip(self.agents, observations)]

    def update_parameters(self, agent_i, sample, updates):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        obs_batch = torch.FloatTensor(obs[agent_i])
        next_obs_batch = torch.FloatTensor(next_obs[agent_i])
        action_batch = torch.FloatTensor(acs[agent_i])
        reward_batch = torch.FloatTensor(rews[agent_i]).unsqueeze(1)
        mask_batch = torch.FloatTensor(
            1 - np.float32(dones[agent_i])).unsqueeze(1)

        """
        Use two Q-functions to mitigate positive bias in the policy improvement step that is known
        to degrade performance of value based methods. Two Q-functions also significantly speed
        up training, especially on harder task.
        """
        qf_in = torch.cat((*obs, *acs), dim=1)
        expected_q1_value, expected_q2_value = curr_agent.critic(qf_in)
        current_q_value = torch.min(expected_q1_value, expected_q2_value)
        new_action, log_prob, _, mean, log_std = curr_agent.policy.sample(
            obs_batch)

        if curr_agent.policy_type == "Gaussian":
            if curr_agent.automatic_entropy_tuning:
                # Alpha Loss
                alpha_loss = -(curr_agent.log_alpha * (log_prob +
                                                       curr_agent.target_entropy).detach()).mean()
                curr_agent.alpha_optim.zero_grad()
                alpha_loss.backward()
                curr_agent.alpha_optim.step()
                curr_agent.alpha = curr_agent.log_alpha.exp()
                alpha_logs = curr_agent.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.)
                alpha_logs = curr_agent.alpha  # For TensorboardX logs

            """
            Including a separate function approximator for the soft value can stabilize training.
            """
            vf_in = torch.cat((obs), dim=1)
            expected_value = curr_agent.value(vf_in)
            target_value = curr_agent.value_target(vf_in)
            next_q_value = reward_batch + mask_batch * \
                self.gamma * (target_value).detach()
        else:
            raise NotImplementedError()
            """
            There is no need in principle to include a separate function approximator for the state value.
            We use a target critic network for deterministic policy and eradicate the value value network completely.
            """
            # alpha_loss = torch.tensor(0.)
            # alpha_logs = curr_agent.alpha  # For TensorboardX logs
            # next_state_action, _, _, _, _, = curr_agent.policy.sample(
            #     next_obs_batch)
            # target_critic_1, target_critic_2 = curr_agent.critic_target(
            #     next_obs_batch, next_state_action)
            # target_critic = torch.min(target_critic_1, target_critic_2)
            # next_q_value = reward_batch + mask_batch * \
            #     self.gamma * (target_critic).detach()

        """
        Soft Q-function parameters can be trained to minimize the soft Bellman residual
        JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        ∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
        """
        all_new_acs = []
        for a_i in range(self.nagents):
            if a_i == agent_i:
                all_new_acs.append(new_action)
            else:
                all_new_acs.append(acs[a_i])
        new_qf_in = torch.cat((*obs, *all_new_acs), dim=1)
        q1_value_loss = F.mse_loss(expected_q1_value, next_q_value)
        q2_value_loss = F.mse_loss(expected_q2_value, next_q_value)
        q1_new, q2_new = curr_agent.critic(new_qf_in)
        expected_new_q_value = torch.min(q1_new, q2_new)

        if curr_agent.policy_type == "Gaussian":
            """
            Including a separate function approximator for the soft value can stabilize training and is convenient to 
            train simultaneously with the other networks
            Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
            JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - α * log π(at|st)]))^2]
            ∇JV = ∇V(st)(V(st) - Q(st,at) + (α * logπ(at|st)))
            """
            next_value = expected_new_q_value - (curr_agent.alpha * log_prob)
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
        policy_loss = ((curr_agent.alpha * log_prob) -
                       expected_new_q_value).mean()

        # Regularization Loss
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()

        policy_loss += mean_loss + std_loss

        curr_agent.critic_optim.zero_grad()
        q1_value_loss.backward()
        curr_agent.critic_optim.step()

        curr_agent.critic_optim.zero_grad()
        q2_value_loss.backward()
        curr_agent.critic_optim.step()

        if curr_agent.policy_type == "Gaussian":
            curr_agent.value_optim.zero_grad()
            value_loss.backward()
            curr_agent.value_optim.step()
        else:
            raise ValueError("Should not be here")
            # value_loss = torch.tensor(0.)

        curr_agent.policy_optim.zero_grad()
        policy_loss.backward()
        curr_agent.policy_optim.step()

        """
        We update the target weights to match the current value function weights periodically
        Update target parameter after every n(args.target_update_interval) updates
        """
        if updates % self.target_update_interval == 0 and curr_agent.policy_type == "Deterministic":
            soft_update(curr_agent.critic_target, curr_agent.critic, self.tau)

        elif updates % self.target_update_interval == 0 and curr_agent.policy_type == "Gaussian":
            soft_update(curr_agent.value_target, curr_agent.value, self.tau)

        return torch.mean(current_q_value).item(), value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_logs

    def distill(self, num_distill, batch_size, replay_buffer, temperature=0.01, tau=0.01, pass_actor=False, pass_critic=False):
        if pass_actor and pass_critic:
            return 0

        # Repeat multiple times
        for i in range(num_distill):
            # Get samples
            sample = replay_buffer.sample(batch_size, to_gpu=False)
            obs, acs, _, _, _ = sample

            # Get distributions for observations
            all_actor_dists = []
            distilled_dists = []
            for agent, ob in zip(self.agents, obs):
                with torch.no_grad():
                    all_actor_dists.append(agent.get_distribution(ob))

                distilled_dists.append(
                    self.distilled_agent.get_distribution(ob))

            # Find critic outputs for each agent + distilled
            # Mix input to critic by shuffling
            all_values = []
            all_critic_logits = []
            distilled_values = []
            distilled_critic_logits = []
            for (crit, value) in zip(self.critics, self.values):
                # Inputs for agents
                vf_in = torch.cat((obs), dim=1)
                qf_in = torch.cat((*obs, *acs), dim=1)

                # Inputs for distilled agent
                # Shuffle order of observations/actions
                dist_qf_in = list(zip(obs, acs))
                np.random.shuffle(dist_qf_in)
                dist_obs, dist_acs = zip(*dist_qf_in)

                # Create input for distilled agent
                vf_in_distilled = torch.cat((dist_obs), dim=1)
                qf_in_distilled = torch.cat((*dist_obs, *dist_acs), dim=1)

                # Get critic/value outputs
                val = value(vf_in)
                dist_val = self.distilled_agent.value(vf_in_distilled)

                crit_logit_1, crit_logit_2 = crit(qf_in)
                dist_crit_logit_1, dist_crit_logit_2 = self.distilled_agent.critic(
                    qf_in_distilled)

                # Add to arrays
                all_values.append(val)
                distilled_values.append(dist_val)

                all_critic_logits.append(torch.min(crit_logit_1, crit_logit_2))
                distilled_critic_logits.append(
                    torch.min(dist_crit_logit_1, dist_crit_logit_2))

            for j, agent in enumerate(self.agents):
                if not pass_actor:
                    # Distill agent
                    self.distilled_agent.policy_optim.zero_grad()

                    kl_loss = kl_divergence(
                        p=all_actor_dists[j],
                        q=distilled_dists[j])
                    kl_loss = kl_loss.sum()
                    kl_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.distilled_agent.policy.parameters(), 0.5)
                    self.distilled_agent.policy_optim.step()

                if not pass_critic:
                    # Distill critic
                    self.distilled_agent.critic_optim.zero_grad()

                    target = all_critic_logits[j].detach()
                    student = distilled_critic_logits[j]
                    loss = F.mse_loss(student, target)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.distilled_agent.critic.parameters(), 0.5)
                    self.distilled_agent.critic_optim.step()

                    # Distill value
                    self.distilled_agent.value_optim.zero_grad()

                    target = all_values[j].detach()
                    student = distilled_values[j]
                    loss = F.mse_loss(student, target)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.distilled_agent.value.parameters(), 0.5)
                    self.distilled_agent.value_optim.step()

        # Update student parameters
        for a in self.agents:
            if not pass_actor:
                a.policy.load_state_dict(
                    self.distilled_agent.policy.state_dict())

            if not pass_critic:
                a.critic.load_state_dict(
                    self.distilled_agent.critic.state_dict())
                a.value.load_state_dict(
                    self.distilled_agent.value.state_dict())
                a.value_target.load_state_dict(
                    self.distilled_agent.value.state_dict())

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
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

            # Q-network and Value network
            num_in_critic = 0
            num_in_value = 0
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
                num_in_value += oobsp.shape[0]

            for oacsp in env.action_space:
                num_in_critic += oacsp.shape[0]

            agent_init_params.append({
                'agent_id': agent_id,
                'num_in_pol': num_in_pol,
                'num_out_pol': num_out_pol,
                'num_in_critic': num_in_critic,
                'num_in_value': num_in_value})
            agent_id += 1

        init_dict = {
            'gamma': args.gamma, 'tau': args.tau, 'lr': args.lr, 'num_agents': agent_id,
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
