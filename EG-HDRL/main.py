from utils import str2bool, evaluate_policy, evaluate_policy1, evaluate_policy_vspdc, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC import SAC_countinuous
from Docking_controller import VSPDCController
import gymnasium as gym
import os, shutil
import argparse
import torch
import gym
import rospy
import numpy as np          # ← Added to import section

# Gym Environment parameters
ENV_NAME = 'IndustrialDrone-v0'
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Drone-SS, PV1, LLdV2, Humanv4, HCv4, BWv3, BWHv3')
# help=  Just a prompt for command-line users to view, does not affect the program
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2700, help='which model to load')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(2.75e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(10000), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Frequency, in steps')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=128, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')  # Entropy coefficient, adjustable
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    EnvName = ['IndustrialDrone-v0', 'Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['Drone', 'PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    # Initialize ROS node
    rospy.set_param("use_sim_time", True)
    rospy.init_node('drone_env_test', anonymous=True)
    # Build Env
    # env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex])
    # eval_env = gym.make(ENV_NAME)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   # remark: action space【-max,max】

    action_space_high = env.action_space.high # [0.1, 0.1, 0.1, 0.5]
    action_space_low = env.action_space.low # [-0.1, -0.1, -0.1, -0.5]

    opt.max_e_steps = env._max_episode_steps  # Maximum steps per episode in the environment
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
    # Instantiate expert controller
    vspdc = VSPDCController(fx=454.6857718666893,
                            fy=454.6857718666893,
                            cx=424.5,
                            cy=424.5)
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        score = evaluate_policy1(env, action_space_high, action_space_low, agent, turns=500)
        print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
        print("✅ Evaluation completed, data saved to evaluation_logs folder, program exited.")
        # score1 = evaluate_policy_vspdc(env, vspdc, action_space_high, action_space_low, turns=50)
        # print('EnvName:', BrifEnvName[opt.EnvIdex], 'score1:', score1)
        # print("✅ Evaluation completed, data saved to evaluation_logs_vspdc folder, program exited.")
        return  # or exit(0)
    else:
        # ---------- 1 Collect expert data  ----------#
        collect_expert_data(env, vspdc, agent.replay_buffer,
                            action_space_high, action_space_low,
                            target_steps=80000)
        # ---------- 2 Pure BC warm-up ----------
        print("🔧 Behaviour-Cloning warm-up …")
        bc_steps = 50000  # Empirically 3k-10k is sufficient
        for _ in range(bc_steps):
            batch = agent.replay_buffer.sample(opt.batch_size)
            agent.update_actor_bc(batch)  # <-- Call patch
        print("✅ BC warm-up done!")
        # ===== 3 DAgger 2~5 rounds =====
        DAGGER_ROUNDS = 5
        EP_PER_ROUND = 3
        for k in range(DAGGER_ROUNDS):
            print(f"🔄  DAgger round {k + 1}/{DAGGER_ROUNDS}")
            run_dagger_cycle(env, agent.replay_buffer, agent, vspdc,
                             action_space_high, action_space_low,
                             n_episode=EP_PER_ROUND)
            #   —— Perform BC again on the newly added data ——
            bc_steps = 2000  # A small amount is enough
            for _ in range(bc_steps):
                batch = agent.replay_buffer.sample(opt.batch_size)
                agent.update_actor_bc(batch)
        # ---------- 4 Formal SAC offline fine-tuning ----------
        offline_steps = 50000  # SAC + Critic updates
        for _ in range(offline_steps):
            agent.train()  # Still sampling expert data

        # ① Record buffer size at the end of offline stage (pure expert data)
        exp_size_snapshot = agent.replay_buffer.size
        # ---------- 5 Online warm-up: collect only 5000 steps, no training ----------
        print("📥 Collecting 5k online transitions before mixed training …")
        online_collect_target = 5000
        online_collected = 0
        while online_collected < online_collect_target:
            s, _ = env.reset(seed=env_seed)
            env_seed += 1
            # env_seed = 1 # Reset environment seed
            done = False
            while not done and online_collected < online_collect_target:
                a = agent.select_action(s, deterministic=True)  # Sample using current policy
                act = Action_adapter(a, action_space_high, action_space_low)
                s_next, r, dw, tr, _ = env.step(act)
                done = dw or tr
                agent.replay_buffer.add(s, a, r, s_next, float(done))
                s = s_next
                online_collected += 1
        print("✅ Online warm-up data collected.")
        # ② Write back expert segment length
        agent.replay_buffer.expert_size = exp_size_snapshot
        # ---------- 5 Formal SAC online training ----------
        print("🚀 Start online interactive training")
        total_steps = 0
        while total_steps < opt.Max_train_steps:  # Total training steps 5e5  500000 times
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & train'''
            while not done:
                a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                act = Action_adapter(a, action_space_high, action_space_low)  # act∈[-max,max]
                # print("act:", act)
                s_next, r, dw, tr, info = env.step(act)  # dw: dead&win; tr: truncated
                print("total_steps:",total_steps)
                done = (dw or tr)
                agent.replay_buffer.add(s, a, r, s_next, done)
                s = s_next
                total_steps += 1

                '''train if it's time'''
                # ========== Replace training call part in the online training loop ==========
                if total_steps % opt.update_every == 0:
                    N_exp = agent.replay_buffer.expert_size
                    N_onl = agent.replay_buffer.size - N_exp
                    # ratio = max(0.05, 1.0 - N_onl / (1.5 * N_exp))  # Adaptive decay
                    ratio = max(0.05, 1.0 - N_onl / (1.5 * N_exp))  # Adaptive decay
                    for _ in range(opt.update_every):
                        batch = agent.replay_buffer.sample_mixed(opt.batch_size, ratio)
                        agent.train_on_batch(batch)

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    MAX_RETRY = 10  # Maximum re-evaluation attempts (prevent long hangs)
                    attempt = 0
                    while True:
                        ep_r = evaluate_policy(
                            eval_env,
                            action_space_high,
                            action_space_low,
                            agent,
                            turns=5
                        )
                        # End loop if evaluation is normal or max retries reached
                        if ep_r != -100 or attempt >= MAX_RETRY:
                            break
                        attempt += 1
                        print(f"[Eval] Got sentinel -100 (attempt {attempt}/{MAX_RETRY}), re-evaluating…")
                    # —— Record to TensorBoard & console ——
                    if opt.write:
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    env_name = BrifEnvName[opt.EnvIdex]
                    print(f'EnvName:{env_name}, Steps: {total_steps // 1000}k, Episode Reward:{ep_r}')

                '''save model'''
                if total_steps % opt.save_interval == 0:   #opt.save_interval=100e3=100000
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
        env.close()
        eval_env.close()

def collect_expert_data(env, controller, buffer,
                        high, low,
                        target_steps = 5000):
    """Let VSPDC fly in the environment and fill buffer before target_steps entries"""
    print(f"🚀  Collecting {target_steps} expert transitions ...")
    pbar_cnt = 0
    while buffer.size < target_steps:
        # print("buffer.size:", buffer.size)
        s, _ = env.reset()
        done = False
        while not done and buffer.size < target_steps:
            uav = env.get_expert_inputs()      # (u,v,θ,depth) or None
            if uav is not None:
                u, v, theta, depth = uav
                act = controller.compute_action((u, v, depth, theta))
            else:
                act = np.zeros(4, dtype=np.float32)   # Target lost → Hover
            # Clip to action space
            act = np.clip(act, low, high)
            # Normalize action and write into buffer
            a_norm = Action_adapter_reverse(act, high, low)
            s_next, r, dw, tr, _ = env.step(act)
            done = dw or tr
            buffer.add(s, a_norm, r, s_next, float(done))
            s = s_next

        # Progress hint
        pbar_cnt += 1
        if pbar_cnt % 10 == 0:
            print(f" ↳ buffer size: {buffer.size}/{target_steps}")
    print("✅ expert buffer ready!")


def run_dagger_cycle(
        env, buffer, agent, expert_ctrl,
        action_high, action_low,
        n_episode: int = 2,
        max_step:   int = 800,
        reward_adapter=None      # Unified Reward_adapter can be passed in
):
    """
    One DAgger cycle:
       • Execute the environment with the current policy π
       • Expert labels action a_exp on the same state s
       • Append (s, a_exp, r, s', done) to the replay buffer
    """
    for ep in range(n_episode):
        s, _ = env.reset()
        for t in range(max_step):
            # ---------- 1.  RL policy action ----------
            a_rl = agent.select_action(s, deterministic=False)
            act  = Action_adapter(a_rl, action_high, action_low)

            # ---------- 2.  Step forward in the environment ----------
            s_next, r, dw, tr, _ = env.step(act)
            done = dw or tr

            # ---------- 3.  Get expert-labeled action ----------
            uav = env.get_expert_inputs()          # (u,v,θ,d) or None
            if uav is None:
                # Target missing in current frame → use hover action 0 as expert label
                a_exp = np.zeros_like(act)
            else:
                u, v, theta, depth = uav
                a_exp = expert_ctrl.compute_action((u, v, depth, theta))

            # Ensure action is in SAC space
            a_exp = np.clip(a_exp, action_low, action_high)
            a_exp_norm = Action_adapter_reverse(a_exp, action_high, action_low)

            # ---------- 4.  Write into replay buffer ----------
            # If reward normalization function exists, keep consistent with training phase
            if reward_adapter:
                r = reward_adapter(r)

            buffer.add(s, a_exp_norm, r, s_next, float(done))

            # ---------- 5.  Iterate ----------
            s = s_next
            if done:
                break

        print(f"DAgger episode {ep+1}/{n_episode}  |  steps: {t+1}")

    print("✅  DAgger cycle finished – new expert-labelled data appended.")


if __name__ == '__main__':
    main()
