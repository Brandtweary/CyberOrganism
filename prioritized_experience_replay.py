import numpy as np
from collections import namedtuple
from typing import List, Tuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedExperienceReplay:
    def __init__(self, capacity: int, batch_size: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, experience: Experience) -> None:
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self) -> Tuple[List[Experience], List[int], np.ndarray]:
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            priorities.append(priority)
            batch.append(experience)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.capacity * sampling_probabilities) ** -self.beta
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, idxs, weights

    def update_priorities(self, idxs: List[int], priorities: np.ndarray) -> None:
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
