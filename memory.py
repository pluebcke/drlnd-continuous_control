from collections import deque
import random

import numpy as np
import torch

class ReplayMemory:
    """ The replay memory stores tuples of state, action, reward, next action and a flag if the episode is done
        The sample method takes batch_size as an input argument and returns this many samples from the buffer
    """
    def __init__(self, device, maxLen):
        self.data = deque(maxlen=maxLen)
        self.device = device
        self.range = np.arange(0, maxLen, 1)
        return

    def add(self, sample):
        self.data.append(sample)
        return

    def sample_batch(self, batch_size):
        states = []
        actions = []
        rewards = []
        last_states = []
        dones = []

        experiences = random.sample(self.data, k=batch_size)

        for experience in experiences:
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            last_states.append(experience.last_state)
            dones.append(experience.done)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1).to(self.device)
        last_states = torch.from_numpy(np.array(last_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.float32)).float().unsqueeze(-1).to(self.device)

        return tuple((states, actions, rewards, last_states, dones))

    def number_samples(self):
        return len(self.data)
