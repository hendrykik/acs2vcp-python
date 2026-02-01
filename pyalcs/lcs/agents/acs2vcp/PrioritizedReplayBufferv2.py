import numpy as np
from lcs.agents.acs2er import ReplayMemorySample


class PrioritizedReplayBufferv2:
    """
    Optimized Prioritized Replay Buffer with cached probabilities.
    
    Improvements over v1:
    - Lazy probability computation (only recomputes when dirty)
    - Cached normalized probabilities for faster sampling
    """
    
    def __init__(self, max_size, batch_size, eps=1e-6, T=1.0):
        self.max_size = max_size
        self.batch_size = batch_size
        self.eps = eps
        self.T = T
        self.sigma2_max = 0.0
        
        # Use lists for O(1) indexed access
        self.buffer = []
        self.priorities = []
        self.sigmas = []
        
        # Cached probabilities for faster sampling
        self._cached_probs = None
        self._probs_dirty = True

    def add(self, sample: ReplayMemorySample, sigma: float, priority: float):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
            self.sigmas.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append(sample)
        self.sigmas.append(sigma)
        self.priorities.append(priority)
        self._probs_dirty = True

    def _ensure_probs_cached(self):
        """Lazily compute and cache normalized probabilities."""
        if self._probs_dirty or self._cached_probs is None or len(self._cached_probs) != len(self.buffer):
            probs = np.array(self.priorities, dtype=np.float64)
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                # Uniform distribution if all priorities are zero
                probs = np.ones(len(probs)) / len(probs)
            self._cached_probs = probs
            self._probs_dirty = False

    def sample(self):
        self._ensure_probs_cached()
        probs = self._cached_probs

        if len(self.buffer) < self.batch_size:
            indices = list(range(len(self.buffer)))
            weights = [1 / (len(self.buffer) * probs[i]) for i in indices]
            return list(self.buffer), indices, weights

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = [1 / (len(self.buffer) * probs[i]) for i in indices]
        return samples, indices.tolist(), weights

    def update_priorities(self, indices, sigmas, new_priorities):
        for i, s, p in zip(indices, sigmas, new_priorities):
            if 0 <= i < len(self.priorities):
                self.priorities[i] = p
                self.sigmas[i] = s
        self._probs_dirty = True
    
    def __len__(self):
        return len(self.buffer)

