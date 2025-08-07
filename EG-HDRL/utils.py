import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
from torch.distributions import Normal
from Docking_controller import VSPDCController

# Instantiate expert controller
vspdc = VSPDCController(fx=454.6857718666893,
                        fy=454.6857718666893,
                        cx=424.5,
                        cy=424.5)

def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape)

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # clamp may harm learning
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


# reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r


def Action_adapter(a, action_space_high, action_space_low):
    # Adapt from [-1,1] to [action_space_low, action_space_high]
    return action_space_low + (a + 1.0) * (action_space_high - action_space_low) / 2.0


def Action_adapter_reverse(act, action_space_high, action_space_low):
    # Adapt from [action_space_low, action_space_high] to [-1,1]
    return 2.0 * (act - action_space_low) / (action_space_high - action_space_low) - 1.0


def evaluate_policy(env, action_space_high, action_space_low, agent, turns=50):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, action_space_high, action_space_low)
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores / turns)


def evaluate_policy1(env, action_space_high, action_space_low, agent, turns=10, env_seed=0):
    total_scores = 0
    per_episode_scores = []  # ← Added: record per-episode score

    # Create directory to save evaluation data
    save_dir = "evaluation_logs111"
    os.makedirs(save_dir, exist_ok=True)

    for j in range(turns):
        s, _ = env.reset(seed=env_seed)
        env_seed += 1
        done = False
        episode_data = []  # data for each episode
        episode_score = 0  # ← score for current episode

        while not done:
            a = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, action_space_high, action_space_low)
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            # De-normalize observation
            raw_obs = (env.obs_high + env.obs_low) / 2 + s_next * (env.obs_high - env.obs_low) / 2
            x, y, theta, distance, rol, pitch, yaw, vx, vy, vz, wz = raw_obs

            # Get UAV world coordinates (local_position/pose)
            try:
                from geometry_msgs.msg import PoseStamped
                import rospy
                pose_msg = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped, timeout=2.0)
                x_world = pose_msg.pose.position.x
                y_world = pose_msg.pose.position.y
                z_world = pose_msg.pose.position.z
            except:
                x_world, y_world, z_world = None, None, None

            episode_data.append([
                env_seed, x, y, theta, distance, rol, pitch, yaw, vx, vy, vz, wz,
                x_world, y_world, z_world, r
            ])

            total_scores += r
            episode_score += r  # ← accumulate episode score
            s = s_next

        per_episode_scores.append(episode_score)  # ← record this episode's score

        print(f"Episode {j+1} | Score: {episode_score:.2f}")

        # Save current episode data
        save_path = os.path.join(save_dir, f"episode_{j+1}.csv")
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['env_seed','x', 'y', 'theta', 'distance', 'rol', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'wz',
                             'x_world', 'y_world', 'z_world', 'r'])  # header
            writer.writerows(episode_data)

    # ✅ Save score log
    scores_log_path = os.path.join(save_dir, "scores_log.csv")
    with open(scores_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score"])
        for i, score in enumerate(per_episode_scores, 1):
            writer.writerow([i, score])
        writer.writerow(["Average", total_scores / turns])  # ← add average

    return int(total_scores / turns)

def evaluate_policy_vspdc(env, vspdc, action_space_high, action_space_low, turns=10, env_seed=0):
    total_scores = 0
    per_episode_scores = []

    # Create directory to save evaluation data
    # save_dir = "evaluation_logs_vspdc"
    save_dir = "evaluation_logs_v4_vspdc"
    os.makedirs(save_dir, exist_ok=True)

    for j in range(turns):
        s, _ = env.reset(seed=env_seed)
        env_seed += 1
        done = False
        episode_data = []
        episode_score = 0

        while not done:
            uav = env.get_expert_inputs()
            if uav is not None:
                u, v, theta, depth = uav
                act = vspdc.compute_action((u, v, depth, theta))
            else:
                act = np.zeros(4, dtype=np.float32)

            act = np.clip(act, action_space_low, action_space_high)
            s_next, r, dw, tr, _ = env.step(act)
            done = (dw or tr)

            raw_obs = (env.obs_high + env.obs_low) / 2 + s_next * (env.obs_high - env.obs_low) / 2
            x, y, theta, distance, rol, pitch, yaw, vx, vy, vz, wz = raw_obs

            try:
                from geometry_msgs.msg import PoseStamped
                import rospy
                pose_msg = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped, timeout=2.0)
                x_world = pose_msg.pose.position.x
                y_world = pose_msg.pose.position.y
                z_world = pose_msg.pose.position.z
            except:
                x_world, y_world, z_world = None, None, None

            episode_data.append([
                env_seed, x, y, theta, distance, rol, pitch, yaw, vx, vy, vz, wz,
                x_world, y_world, z_world, r
            ])

            total_scores += r
            episode_score += r
            s = s_next

        per_episode_scores.append(episode_score)

        print(f"Episode {j + 1} | Score: {episode_score:.2f}")
        save_path = os.path.join(save_dir, f"episode_{j + 1}.csv")
        with open(save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['env_seed','x', 'y', 'theta', 'distance', 'rol', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'wz',
                             'x_world', 'y_world', 'z_world', 'r'])
            writer.writerows(episode_data)

    # Write per-episode scores and average score
    scores_log_path = os.path.join(save_dir, "scores_log.csv")
    with open(scores_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score"])
        for i, score in enumerate(per_episode_scores, 1):
            writer.writerow([i, score])
        writer.writerow(["Average", total_scores / turns])

    return int(total_scores / turns)


def str2bool(v):
    '''Convert str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
