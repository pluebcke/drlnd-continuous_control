from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as functional
import torch.optim as optim

from model_td3 import Actor, Critic
from memory import ReplayMemory

Experience = namedtuple('Experience', 'state action reward last_state, done')


class T3DAgent:
    def __init__(self, env, brain, brain_name, device, settings):
        self.env = env
        self.brain_name = brain_name
        self.device = device
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        state_size = states.shape[1]
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = settings['batch_size']

        # Initialize actor local and target networks
        self.actor_local = Actor(state_size, action_size, settings['actor_settings']).to(device)
        self.actor_target = Actor(state_size, action_size, settings['actor_settings']).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=settings['lr_actor'])

        # Initialize critic networks
        self.critic_local = Critic(state_size, action_size, settings['critic_settings']).to(device)
        self.critic_target = Critic(state_size, action_size, settings['critic_settings']).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=settings['lr_critic'])

        # Save some of the settings into class member variables
        self.pretrain_steps = settings['pretrain_steps']
        self.number_steps = settings['number_steps']
        self.gamma = settings['gamma']
        self.tau = settings['tau']

        self.action_noise = settings['action_noise']
        self.action_clip = settings['action_clip']
        self.target_action_noise = settings['target_action_noise']
        self.target_noise_clip = settings['target_noise_clip']
        self.optimize_every = settings['optimize_critic_every']

        # Initialize replay memory and episode generator
        self.memory = ReplayMemory(device, settings['buffer_size'])
        self.generator = self.play_episode()

        self.number_steps = 0
        return

    def get_action_noise(self):
        return self.action_noise

    def set_action_noise(self, std):
        self.action_noise = std
        return

    def pretrain(self):
        # The idea of using a pretrain phase before starting regular episodes
        # is from https://github.com/whiterabbitobj/Continuous_Control/
        print("Random sampling of " + str(self.pretrain_steps) + " steps")
        env = self.env
        brain_name = self.brain_name
        env_info = env.reset(train_mode=True)[brain_name]
        number_agents = env_info.vector_observations.shape[0]
        for _ in range(self.pretrain_steps):
            actions = []
            states = env_info.vector_observations
            for _ in range(number_agents):
                actions.append(np.random.uniform(-1, 1, self.action_size))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(Experience(state, action, reward, next_state, done))
            if np.any(dones):
                env_info = env.reset(train_mode=True)[brain_name]

    def play_episode(self):
        # The idea of generating episodes in an "experience generator" is from
        # "Deep Reinforcement Learning Hands-On" by Maxim Lapan

        print("Starting episode generator")
        # Initialize the environment
        env = self.env
        brain_name = self.brain_name
        env_info = env.reset(train_mode=True)[brain_name]
        # Initialize episode_rewards and get the first state
        episode_rewards = []
        # Run episode step by step
        while True:
            states = env_info.vector_observations
            with torch.no_grad():
                actions = self.actor_local.forward(
                    torch.from_numpy(states).type(torch.FloatTensor).to(self.device)).cpu().detach().numpy()
                actions += self.action_noise * np.random.normal(size=actions.shape)
                actions = np.clip(actions, -self.action_clip, self.action_clip)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            episode_rewards.append(rewards)

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(Experience(state, action, reward, next_state, done))
            if np.any(dones):
                agent_reward = np.sum(episode_rewards, axis=0)
                std_reward = np.std(agent_reward)
                mean_reward = np.mean(agent_reward)
                episode_rewards = []
                env_info = env.reset(train_mode=True)[brain_name]
                yield mean_reward, std_reward
            else:
                yield -1, -1

    def take_step(self):
        return next(self.generator)

    def learn(self):
        self.number_steps += 1
        if self.memory.number_samples() <= self.batch_size:
            return
        # states, actions, rewards, next states, done
        s0, a0, r, s1, d = self.memory.sample_batch(self.batch_size)
        critic_loss_a, critic_loss_b = self.optimize_critic(s0, a0, r, s1, d)
        actor_loss = self.optimize_actor(s0)

        return actor_loss, critic_loss_a, critic_loss_b

    def optimize_actor(self, s0):
        # Calc policy loss
        if self.number_steps % self.optimize_every == 0:
            a0_pred = self.actor_local(s0)
            actor_loss = -self.critic_local.get_qa(s0, a0_pred).mean()
            # Update actor nn
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # slow update
            self.slow_update(self.tau)
            return -actor_loss.cpu().detach().numpy()
        return 0

    def optimize_critic(self, s0, a0, r, s1, d):
        # The ideas of adding noise to the next state a1 as well as the critic loss, that takes q1_expected and
        # q2_expected as arguments at the same time, are from the implementation of the authors of the TD3 manuscript
        # at https://github.com/sfujim/TD3/
        with torch.no_grad():
            # calc critic loss
            noise = torch.randn_like(a0).to(self.device)
            noise = noise * torch.tensor(self.target_action_noise).expand_as(noise).to(self.device)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            a1 = (self.actor_target(s1) + noise).clamp(-self.action_clip, self.action_clip)
            qa_target, qb_target = self.critic_target(s1, a1)
            q_target = torch.min(qa_target, qb_target)
            q_target = r + self.gamma * (1.0 - d) * q_target
        qa_expected, qb_expected = self.critic_local(s0, a0)
        critic_loss_a = functional.mse_loss(qa_expected, q_target)
        critic_loss_b = functional.mse_loss(qb_expected, q_target)
        critic_loss = critic_loss_a + critic_loss_b
        # Update critic nn
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        return critic_loss_a.cpu().detach().numpy(), critic_loss_b.cpu().detach().numpy()

    def slow_update(self, tau):
        for target_par, local_par in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_par.data.copy_(tau * local_par.data + (1.0 - tau) * target_par.data)
        for target_par, local_par in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_par.data.copy_(tau * local_par.data + (1.0 - tau) * target_par.data)
        return

    def load_nets(self, actor_file_path, critic_file_path):
        self.actor_local.load_state_dict(torch.load(actor_file_path))
        self.actor_local.eval()
        self.critic_local.load_state_dict(torch.load(critic_file_path))
        self.critic_local.eval()
        return

    def save_nets(self, model_save_path):
        actor_path = model_save_path + "_actor_net.pt"
        torch.save(self.actor_local.state_dict(), actor_path)
        critic_path = model_save_path + "_critic_net.pt"
        torch.save(self.critic_local.state_dict(), critic_path)
        return