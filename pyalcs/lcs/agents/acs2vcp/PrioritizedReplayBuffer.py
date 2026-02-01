import numpy as np
from lcs.agents.acs2er import ReplayMemorySample

class PrioritizedReplayBuffer:
    def __init__(self, max_size, batch_size, eps=1e-6, T=1.0):
        self.buffer = []
        self.priorities = []
        self.sigmas = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.eps = eps
        self.T = T
        self.sigma2_max = 0.0

    def add(self, sample: ReplayMemorySample, sigma: float, priority: float):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
            self.sigmas.pop(0)
            self.priorities.pop(0)
        self.buffer.append(sample)
        self.sigmas.append(sigma)
        self.priorities.append(priority)

    def sample(self):
        probs = np.array(self.priorities, dtype=np.float64)
        probs /= probs.sum()

        if len(self.buffer) < self.batch_size:
            indices = [i for i in range(len(self.buffer))]
            weights = [1 / (len(self.buffer) * probs[i]) for i in indices]
            return self.buffer, indices, weights

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = [1 / (len(self.buffer) * probs[i]) for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, sigmas, new_priorities):
        for i, s, p in zip(indices, sigmas, new_priorities):
            self.priorities[i] = p
            self.sigmas[i] = s