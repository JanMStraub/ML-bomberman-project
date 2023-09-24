# -*- coding: utf-8 -*-

"""
Reinforcement Learning agent for the game Bomberman.
@author: Christian Teutsch, Jan Straub
"""
from typing import List
from collections import deque

import pickle
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import numpy as np
import events as e
from settings import COLS, ROWS
from .callbacks import state_to_features, ACTIONS
from .q_network import DQN
from .replay_memory import ReplayMemory, Transition
from .helper import check_for_loop, is_valid_movement_action, bomb_at_spawn
from .helper import closer_to_coin, destroy_crate_action_reward, bomb_evaded


# Hyper parameters
GAMMA = 0.9
BATCH_SIZE = 512
MAT_SIZE = COLS * ROWS
HIDDEN_SIZE = 2048 # 1734
STEP_SIZE = 10000
LEARNING_RATE = 0.3
DROPOUT = 0.6

# Events
LOOP = "LOOP"
VALID_ACTION = "VALID_ACTION"
INVALID_ACTION = "INVALID_ACTION"
BOMB_AT_SPAWN = "BOMB_AT_SPAWN"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
DESTROY_CRATE = "DESTROY_CRATE"
BOMB_EVADED = "BOMB_EVADED"


def setup_training(self):
    """
    Initialize the agent for training.
    """
    self.logger.info("Setting up the training environment.")
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.valid_actions = []
    self.closest_coin_position = None
    self.my_bomb_position = None

    # Initialize the policy network and the target network
    self.policy_net = DQN(MAT_SIZE,
                          len(ACTIONS),
                          HIDDEN_SIZE,
                          DROPOUT)
    self.target_net = DQN(MAT_SIZE,
                          len(ACTIONS),
                          HIDDEN_SIZE,
                          DROPOUT)

    # Load the initial weights of the target network
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.SGD(self.policy_net.parameters(),
                               lr = LEARNING_RATE)
    self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                         step_size = STEP_SIZE,
                                         gamma = GAMMA)
    self.memory = ReplayMemory(STEP_SIZE)


def game_events_occurred(self,
                         old_game_state: dict,
                         self_action: str,
                         new_game_state: dict,
                         events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """

    self.logger.debug(f'Encountered game event(s) \
                      {", ".join(map(repr, events))} \
                      in step {new_game_state["step"]}')

    # Gather information about the game state
    self.active_bomb_positions = [xy for xy, _ in new_game_state['bombs']]
    self.others = [xy for (_, _, _, xy) in new_game_state['others']]
    field_shape = new_game_state['field'].shape
    self.bomb_map = np.full(field_shape, 5)

    for (xb, yb), t in new_game_state['bombs']:
        for i, j in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if 0 < i < field_shape[0] and 0 < j < field_shape[1]:
                self.bomb_map[i, j] = min(self.bomb_map[i, j], t)

    if self_action == "BOMB":
        self.my_bomb_position = old_game_state["self"][3]

    check_conditions(self,
                     old_game_state,
                     new_game_state,
                     events,
                     self_action)

    self.memory.push(state_to_features(old_game_state),
                     self_action,
                     state_to_features(new_game_state),
                     reward_from_events(self,
                                        events))
    optimize_model(self,
                   self_action)

    self.valid_actions = []

def check_conditions(self,
                     old_game_state: dict,
                     new_game_state: dict,
                     events: List[str],
                     self_action: str):
    """
    Check specific conditions and append corresponding events.
    """

    if check_for_loop(self,
                      new_game_state):
        self.logger.debug("Event: LOOP")
        events.append(LOOP)
    
    if is_valid_movement_action(self,
                                old_game_state,
                                self_action):
        self.logger.debug("Event: VALID_ACTION")
        events.append(VALID_ACTION)
    else:
        self.logger.debug("Event: INVALID_ACTION")
        events.append(INVALID_ACTION)

    if bomb_at_spawn(old_game_state,
                     new_game_state):
        self.logger.debug("Event: BOMB_AT_SPAWN")
        events.append(BOMB_AT_SPAWN)
    
    if closer_to_coin(self,
                      old_game_state,
                      new_game_state):
        self.logger.debug("Event: CLOSER_TO_COIN")
        events.append(CLOSER_TO_COIN)
    
    if destroy_crate_action_reward(new_game_state,
                                   self_action):
        self.logger.debug("Event: DESTROY_CRATE")
        events.append(DESTROY_CRATE)

    if bomb_evaded(new_game_state):
        self.logger.debug("Event: BOMB_EVADED")
        events.append(BOMB_EVADED)


def end_of_round(self,
                 last_game_state: dict,
                 last_action: str,
                 events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.memory.push(state_to_features(last_game_state),
                     last_action,
                     None,
                     reward_from_events(self,
                                        events))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net,
                    file)

    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)


def reward_from_events(self,
                       events: List[str]) -> int:
    """
    Calculate the total reward based on a dictionary of event rewards.
    """

    # Define a dictionary to map events to rewards
    event_rewards = {
        e.WAITED: -0.8,
        #e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -0.2,
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0.5,
        e.COIN_COLLECTED: 0.5,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -1,
        e.SURVIVED_ROUND: 1,
        LOOP: -1,
        VALID_ACTION: 0.3,
        INVALID_ACTION: -2,
        BOMB_AT_SPAWN: -10,
        CLOSER_TO_COIN: 0.4,
        DESTROY_CRATE: 0.5,
        BOMB_EVADED: 0.7,
    }

    # Calculate the total reward for the given events
    total_reward = sum(event_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {total_reward} for events {', '.join(events)}")
    total_reward -= 0.1

    return total_reward


def optimize_model(self,
                   action):
    """
    Function performs single optimization step.
    """
    # Check if there are enough samples in the replay memory
    if len(self.memory) < BATCH_SIZE:
        return

    # Sample a batch of transitions from replay memory
    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Filter non-final next states and convert them to a PyTorch tensor
    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state],
        dtype = torch.bool)
    non_final_next_states = torch.from_numpy(
        np.array([s for s in batch.next_state if s is not None],
                 dtype = np.float32)
    )

    # Convert batched states to a PyTorch tensor
    state_batch = torch.from_numpy(np.array(batch.state,
                                            dtype = np.float32))

    # Create a one-hot encoded tensor for the selected actions in the batch
    action_indices = [ACTIONS.index(action) for action in batch.action]
    action_batch = torch.zeros(BATCH_SIZE,
                               len(ACTIONS),
                               dtype = torch.float32)
    action_batch[range(BATCH_SIZE), action_indices] = 1

    # Compute Q-values for the selected actions in the current policy network
    state_action_values = self.policy_net(
        state_batch).gather(1,torch.tensor(action_indices).unsqueeze(1))

    # Compute the maximum Q-values for non-final next states using the target network
    next_state_values = torch.zeros(BATCH_SIZE,
                                    dtype = torch.float32)
    next_state_values[non_final_mask] = self.target_net(
        non_final_next_states).max(1).values.detach()

    # Compute the expected Q-values (target) using the Bellman equation
    reward_batch = torch.tensor(batch.reward,
                                dtype = torch.float32)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the loss using the Huber loss (smooth L1 loss)
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model by backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.scheduler.step()
