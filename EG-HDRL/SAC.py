from utils import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class SAC_countinuous():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.001

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

		if self.adaptive_alpha:
			self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return a.cpu().numpy()[0]


	def train_on_batch(self, batch):
		s, a, r, s_next, dw = batch
		# ----------------------------- Update Q Net -----------------------------
		with torch.no_grad():
			a_next, logp_next = self.actor(s_next, False, True)
			target_q1, target_q2 = self.q_critic_target(s_next, a_next)
			target_q = torch.min(target_q1, target_q2)
			target_q = r + (~dw) * self.gamma * (target_q - self.alpha * logp_next)

		current_q1, current_q2 = self.q_critic(s, a)
		q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
		self.q_critic_optimizer.zero_grad();
		q_loss.backward();
		self.q_critic_optimizer.step()
		# ----------------------------- Update Actor Net --------------------------
		for p in self.q_critic.parameters(): p.requires_grad = False
		a_new, logp = self.actor(s, False, True)
		q1_pi, q2_pi = self.q_critic(s, a_new)
		a_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()
		self.actor_optimizer.zero_grad();
		a_loss.backward();
		self.actor_optimizer.step()
		for p in self.q_critic.parameters(): p.requires_grad = True
		# ----------------------------- Update alpha ------------------------------
		if self.adaptive_alpha:
			alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad();
			alpha_loss.backward();
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()
		# ----------------------------- Soft Target -------------------------------
		with torch.no_grad():
			for p, p_t in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

	def train(self):
		batch = self.replay_buffer.sample(self.batch_size)
		self.train_on_batch(batch)

	# ======================================================================
	# ★★★  Behaviour Cloning (BC) warm-up interface  ★★★
	# ======================================================================
	def _actor_forward(self, s):
		"""For BC / DAgger reuse, return actor output action tensor"""
		a_pred, _ = self.actor(s, deterministic=True, with_logprob=False)
		return a_pred

	def update_actor_bc(self, batch):
		"""
         Perform one MSE behaviour cloning update on the policy network using expert action a_exp.
        batch = (s, a_exp, r, s_next, dw) —— same as replay_buffer.sample() output.
        """
		s, a_exp, _, _, _ = batch
		a_pred = self._actor_forward(s)
		bc_loss = F.mse_loss(a_pred, a_exp)

		self.actor_optimizer.zero_grad()
		bc_loss.backward()
		self.actor_optimizer.step()

		return bc_loss.item()  # Convenient for printing in the main loop

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		# Only insert one timestep of data each time
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size # Once full, start overwriting from the beginning
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

	# === Add inside ReplayBuffer class ===
	def sample_mixed(self, batch_size, expert_portion):
		"""
		Sample expert data according to expert_portion, and the rest from the latest online data
		expert_portion: 0.0~1.0
		"""
		# 1) Define "expert segment" as the first self.expert_size entries in the buffer
		exp_size = int(self.expert_size)           # self.expert_size needs to be set by the main program
		onl_size = self.size - exp_size

		n_exp = int(batch_size * expert_portion)
		n_onl = batch_size - n_exp

		# Boundary check
		n_exp = min(n_exp, exp_size)
		n_onl = min(n_onl, onl_size)
		if n_exp + n_onl < batch_size:  # If insufficient, fill all randomly
			n_exp = min(exp_size, batch_size)
			n_onl = batch_size - n_exp

		# Sample indices
		idx_exp = torch.randint(0, exp_size, (n_exp,), device=self.dvc) if n_exp else torch.empty(0, dtype=torch.long, device=self.dvc)
		idx_onl = torch.randint(exp_size, exp_size+onl_size, (n_onl,), device=self.dvc) if n_onl else torch.empty(0, dtype=torch.long, device=self.dvc)
		ind = torch.cat((idx_exp, idx_onl), dim=0)

		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
