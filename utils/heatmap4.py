import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from torch.autograd import Variable
from utils.misc import onehot_from_logits

lndmrk_poses = np.array([[0.75, 0.75], [-0.75, 0.75], [-0.75, -0.75], [0.75, -0.75]])
default_agent_poses = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
flipped_order = [3,0,1,2]
flipped_agent_poses = [default_agent_poses[i] for i in flipped_order]
color = {0: 'b', 1: 'r', 2: 'g', 3: [0.65, 0.65, 0.65]}
num_arrows = 21

def get_observations(agent_poses):
    obs_n = []

    for i, agent_pos in enumerate(agent_poses):
        entity_pos = []
        for lndmrk_pos in lndmrk_poses:
            entity_pos.append(lndmrk_pos - agent_pos)
        other_pos = []
        comm = []
        for j, agent_pos_2 in enumerate(agent_poses):
            if i == j: continue
            comm.append(np.array([0, 0]))
            other_pos.append(agent_pos_2 - agent_pos)
        entity_pos = sorted(entity_pos, key=lambda pos: np.arctan2(pos[1], pos[0]))
        other_pos = sorted(other_pos, key=lambda pos: np.arctan2(pos[1], pos[0]))
        obs_n.append(np.concatenate([np.array([0, 0])] + [agent_pos] + entity_pos + other_pos + comm))

    return np.array([obs_n])

def add_arrows(axes, delta_dict, arrow_color="black", q_vals = None, rescale=False):
    max_delta = max(delta_dict.values(), key=(lambda key: np.linalg.norm(key)))
    max_delta_size = np.linalg.norm(max_delta)


    for pos, delta in delta_dict.items():
        if rescale:
            delta = delta/max_delta_size*0.15
        else:
            delta = delta/np.linalg.norm(delta)*0.06

        axes.arrow(pos[0], pos[1], delta[0], delta[1], length_includes_head=True, head_width=0.018, color=arrow_color)

def add_contours(axes, q_vals, fig):
    X = np.linspace(-1, 1, num_arrows)
    Y = np.linspace(-1, 1, num_arrows)
    Z = np.empty((num_arrows, num_arrows))

    for key in q_vals.keys():
        x_index = np.where(X==key[0])[0][0]
        y_index = np.where(Y==key[1])[0][0]
        Z[y_index, x_index] = q_vals[key]
    CS = axes.contourf(X, Y, Z, 25)
    fig.colorbar(CS, ax=axes, shrink=0.9)

def heatmap4(maddpg, title="Agent Policies", save=False):
    fig, axes = plt.subplots(4, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    fig2, axes2 = plt.subplots(4, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    titles = [["Blue Agent, State 1", "Blue Agent, State 2"], ["Red Agent, State 1", "Red Agent, State 2"], ["Green Agent, State 1", "Green Agent, State 2"], ["Silver Agent, State 1", "Silver Agent, State 2"]]

    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            ax2 = axes2[i, j]
            ax2.set_aspect('equal', 'box')
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(-1, 1)
            ax2.set_title(titles[i][j])

            delta_dict = dict()
            val_dict = dict()

            for x in np.linspace(-1, 1, num_arrows):
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if j == 0:
                        agent_poses = np.copy(default_agent_poses)
                    else:
                        agent_poses = np.copy(flipped_agent_poses)
                    agent_poses[i] = agent_pos


                    obs = get_observations(agent_poses)  # Agent 0 (Blue) and 1 (Red)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                          requires_grad=False)
                                 for k in range(maddpg.nagents)]
                    torch_agent_logits = maddpg.action_logits(torch_obs)
                    torch_agent_onehots = [onehot_from_logits(ac) for ac in torch_agent_logits]
                    action = torch_agent_logits[i].data.numpy()[0]
                    # action = torch_agent_onehots[i].data.numpy()[0]

                    obs = [o.repeat(2,1) for o in torch_obs]
                    act = [a.repeat(2,1) for a in torch_agent_onehots]

                    vf_in = torch.cat((*obs, *act), dim=1)
                    vf_out = maddpg.agents[i].critic(vf_in)

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]
                    val_dict[tuple(agent_pos)] = vf_out.mean()

            add_arrows(ax, delta_dict, arrow_color=color[i], rescale=False, q_vals=val_dict)
            add_contours(ax2, val_dict, fig2)
            for l in range(len(agent_poses)):
                if i == l: continue
                ax.add_artist(plt.Circle(agent_poses[l], 0.1, color=color[l]))
                ax2.add_artist(plt.Circle(agent_poses[l], 0.1, color=color[l]))
            for lndmrk_pos in lndmrk_poses:
                ax.add_artist(plt.Circle(lndmrk_pos, 0.05, color=[0.25, 0.25, 0.25]))
                ax2.add_artist(plt.Circle(lndmrk_pos, 0.05, color=[0.25, 0.25, 0.25]))

    fig.suptitle(title)
    fig2.suptitle(title)

    if save:
        plt.savefig("{}.png".format(title), bbox_inches="tight", dpi=300) 
