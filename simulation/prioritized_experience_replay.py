import numpy as np
from collections import namedtuple, deque
from typing import List, Tuple
import random
import threading
from shared.summary_logger import summary_logger

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
        return idx  # Return the index where the experience was added

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def get_priority(self, idx):
        return self.tree[idx]

class PrioritizedExperienceReplay:
    def __init__(self, capacity: int, batch_size: int, alpha: float = 0.6, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.lock = threading.Lock()
        
        self.recent_buffer = []
        self.recent_buffer_max_size = 20 * batch_size

    def add(self, experience: Experience) -> None:
        with self.lock:
            # Add new experience to the recent buffer
            self.recent_buffer.append(experience)
            
            # If recent buffer exceeds max size, move oldest experience to the tree
            if len(self.recent_buffer) > self.recent_buffer_max_size:
                oldest_experience = self.recent_buffer.pop(0)
                self.tree.add(self.max_priority ** self.alpha, oldest_experience)
                summary_logger.warning("Recent buffer capacity exceeded. Added oldest experience to tree.")

    def sample(self) -> Tuple[List[Experience], List[int]]:
        with self.lock:
            batch = []
            idxs = []

            # First, try to sample up to half the batch size from the tree
            tree_sample_size = min(self.tree.n_entries, self.batch_size // 2)
            self._sample_from_tree(batch, idxs, tree_sample_size)

            # Then, sample from the recent buffer
            recent_sample_size = min(len(self.recent_buffer), self.batch_size - len(batch))
            self._sample_from_recent(batch, idxs, recent_sample_size)

            # If we still need more samples, get them from the tree
            remaining_sample_size = self.batch_size - len(batch)
            if remaining_sample_size > 0:
                self._sample_from_tree(batch, idxs, remaining_sample_size)

        return batch, idxs

    def _sample_from_tree(self, batch: List[Experience], idxs: List[int], n_samples: int) -> None:
        if n_samples == 0:
            return
        segment = self.tree.total() / n_samples
        for i in range(n_samples):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, _, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)

    def _sample_from_recent(self, batch: List[Experience], idxs: List[int], n_samples: int) -> None:
        for _ in range(n_samples):
            if not self.recent_buffer:  # Check if recent buffer is empty
                break
            exp = self.recent_buffer.pop(0)  # Get the oldest experience
            idx = self.tree.add(self.max_priority ** self.alpha, exp)
            batch.append(exp)
            idxs.append(idx)

    def update_priorities(self, idxs: List[int], priorities: np.ndarray) -> None:
        with self.lock:
            for idx, priority in zip(idxs, priorities):
                priority = (priority + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)

    def can_sample(self) -> bool:
        with self.lock:
            return len(self.recent_buffer) >= self.batch_size or self.tree.n_entries >= self.batch_size
    
    def get_tree_size(self) -> int:
        with self.lock:
            return self.tree.n_entries
