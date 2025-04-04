import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import numpy as np
from base.segment_tree import SumSegmentTree, MinSegmentTree
from copy import deepcopy
from utils import utils
import os
import logging

# USING BASE SCRIPTS FROM 1. PLACE 2021 COMPETITION
# https://github.com/anticdimi/laser-hockey 

# @modfied
class ExperienceReplay:
    """
    The ExperienceReplay class implements a base class for an experience replay buffer.

    Parameters
    ----------
    max_size : int
        The variable specifies maximum number of (s, a, r, new_state, done) tuples in the buffer.
    """

    def __init__(self, max_size=100000):
        self._transitions = np.asarray([])
        self._current_idx = 0
        self.size = 0
        self.max_size = max_size
        #logging.basicConfig(level=logging.INFO)  # Set up logging


    @staticmethod
    def clone_buffer(new_buffer, maxsize):
        old_transitions = deepcopy(new_buffer._transitions[:new_buffer.size])
        buffer = UniformExperienceReplay(max_size=maxsize)
        for t in old_transitions:
            buffer.add_transition(t)

        return buffer
    
    @staticmethod
    def clone_buffer_per(new_buffer, maxsize, alpha, beta, beta_end):
        old_transitions = deepcopy(new_buffer._transitions[:new_buffer.size])
        buffer = PrioritizedExperienceReplay(max_size=maxsize, alpha=alpha, beta=beta, beta_end=beta_end)
        for t in old_transitions:
            buffer.add_transition(t)

        return buffer

    def add_transition(self, transitions_new):
        if self.size == 0:
            # Initialize buffer with correct structure
            self._transitions = np.empty(self.max_size, dtype=object)
            self._transitions[self._current_idx] = np.asarray(transitions_new, dtype=object)
            self.size = 1
            self._current_idx = 1 % self.max_size
        else:
            transition_arr = np.asarray(transitions_new, dtype=object)
            self._transitions[self._current_idx] = transition_arr
            self.size = min(self.size + 1, self.max_size)
            self._current_idx = (self._current_idx + 1) % self.max_size


    def preload_transitions(self, path):
        for file in os.listdir(path):
            if file.endswith(".npz"):
                fpath = os.path.join(path, file)

                with np.load(fpath, allow_pickle=True) as d:
                    np_data = d['arr_0'].item()

                    if (
                        # Add a fancy condition
                        True
                    ):
                        transitions = utils.recompute_rewards(np_data, "bober") #'Dimitrije_Antic_-_SAC_ЈУГО')
                        for t in transitions:
                            tr = (
                                t[0],
                                t[1],
                                float(t[3]),
                                t[2],
                                bool(t[4]),
                            )
                            self.add_transition(tr)

        print(f'Preloaded data... Buffer size {self.size}.')

    def sample(self, batch_size):
        raise NotImplementedError("Implement the sample method")


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, max_size=100000):
        super(UniformExperienceReplay, self).__init__(max_size)

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size

        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return self._transitions[indices]

# @strongly modified
# Implemented from https://arxiv.org/pdf/1511.05952
class PrioritizedExperienceReplay(ExperienceReplay):
    """
    Prioritized Experience Replay (PER) implementation.
    
    Math Summary:
    - Priority: p_i = |δ_i| + ε  (δ is TD-error, ε prevents zero)
    - Sampling Probability: P(i) = (p_i^α) / Σ_k (p_k^α)
    - Importance Sampling Weight: w_i = ( (N * P(i)) ^ (-β) ) / max(w)
    """
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_end=0.95, epsilon=1e-3):
        max_size_int = int(max_size)
        super(PrioritizedExperienceReplay, self).__init__(max_size_int)
        self._alpha = alpha
        self._beta_start = beta  # Initial beta value
        self._beta_end = beta_end     # Final beta value
        self._epsilon = epsilon  # Added epsilon to avoid zero priority
        self._max_priority = 1.0  # Initial max priority
        print(f"Creating PER buffer with alpha={alpha}, beta={beta}, beta_end={beta_end}, epsilon={epsilon}")

        # Segment Trees for efficient priority management
        st_capacity = 1
        while st_capacity < max_size:
            st_capacity *= 2
        self._st_sum = SumSegmentTree(st_capacity)
        self._st_min = MinSegmentTree(st_capacity)

    def add_transition(self, transitions_new):
        idx = int(self._current_idx)
        super().add_transition(transitions_new)
        # New transitions get max_priority^alpha (ensures they are sampled at least once)
        priority =(self._max_priority + self._epsilon) ** self._alpha
        self._st_min[idx] = priority
        self._st_sum[idx] = priority
        self._max_priority = max(self._max_priority, priority)

    def _sample_proportionally(self, batch_size):
        indices = []
        p_total = self._st_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform(0, 1) * every_range_len + i * every_range_len
            idx = self._st_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        return np.array(indices)

    def sample(self, batch_size):
        if self.size == 0:
            return {'transitions': np.asarray([]), 'weights': np.asarray([]), 'indices': np.asarray([])}
        batch_size = min(batch_size, self.size)
        indices = self._sample_proportionally(batch_size)
        weights = []

        sum_p = self._st_sum.sum(0, self.size - 1)
        min_p = self._st_min.min(0, self.size - 1) / sum_p
        max_weight = (min_p * self.size) ** (-self._beta)

        for idx in indices:
            p_sample = self._st_sum[idx] / sum_p
            weight = (p_sample * self.size) ** (-self._beta)
            weights.append(weight / max_weight)

        sampled_transitions = self._transitions[indices]
        weights = np.array(weights).reshape(-1, 1)
        indices = np.array(indices).reshape(-1, 1)
        return {
            'transitions': sampled_transitions,
            'weights': weights,
            'indices': indices
        }

    def update_priorities(self, indices, priorities):
        """
        Priorities should be computed as |TD_error| + self._epsilon before passing here.
        """
        for idx, priority in zip(indices, priorities):
            assert priority > 0, "Priority must be positive (add epsilon)."
            assert 0 <= idx < self.size, "Index out of buffer range."
            # Update priority with alpha exponent
            priority_alpha = (priority + self._epsilon) ** self._alpha  # Ensure epsilon is added
            self._st_sum[idx] = priority_alpha
            self._st_min[idx] = priority_alpha
            # Update max_priority if necessary
            self._max_priority = max(self._max_priority, priority + self._epsilon)

    def update_beta(self, step, total_steps):
        """
        Anneal beta from its initial value to 1.0 over the course of training.
        
        Parameters:
        - step: Current training step.
        - total_steps: Total number of training steps.
        """
        fraction = min(step / total_steps, 1.0)  # Ensure fraction <= 1.0
        self._beta = self._beta_start + fraction * (self._beta_end - self._beta_start)
