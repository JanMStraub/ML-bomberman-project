# -*- coding: utf-8 -*-

"""
Reinforcement Learning agent for the game Bomberman.
@author: Christian Teutsch, Jan Straub
"""

from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Replay memory object class.
    """

    def __init__(self,
                 capacity):
        self.memory = deque([],
                            maxlen = capacity)

    def __len__(self):
        return len(self.memory)

    def push(self,
             *args):
        """
        Add entry to memory.
        """
        self.memory.append(Transition(*args))

    def sample(self,
               batch_size):
        """
        Get random entry from memory.
        """
        return random.sample(self.memory,
                             batch_size)
