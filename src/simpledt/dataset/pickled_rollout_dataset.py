import os
import random
import pickle
from typing import Callable, Dict, List
import torch
from torch.utils.data import IterableDataset
from dataclasses import dataclass


class PickledRolloutDataset(IterableDataset):
    def __init__(self, dirs: List[str], seq_len: int = 4, preprocess: Callable = None):
        self.dirs = dirs
        self.seq_len = seq_len
        self.preprocess = preprocess
        self.rollouts = []
        self.transitions = 0
        for directory in self.dirs:
            if os.path.isdir(directory):
                for file in os.listdir(directory):
                    if file.endswith('.pkl'):
                        with open(os.path.join(directory, file), 'rb') as f:
                            rollout = pickle.load(f)
                            self.rollouts.append(rollout)
                            self.transitions += rollout.size

    def get_stats(self) -> Dict:
        return {
            'rollouts': len(self.rollouts),
            'transitions': self.transitions,
        }

    def __iter__(self):
        while True:
            rollout = random.choice(self.rollouts)
            start_index = random.randint(0, rollout.size - self.seq_len)
            end_index = start_index + self.seq_len
            if self.preprocess:
                obs, act = self.preprocess(rollout, start_index, end_index)
            else:
                obs = rollout.observations[start_index:end_index]
                act = rollout.actions[start_index:end_index]
            yield obs, act
