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
from settings import COLS, ROWS, BOMB_POWER
from .callbacks import state_to_features, ACTIONS
from .q_network import DQN
from .replay_memory import ReplayMemory, Transition
from .helper import check_danger_zone, find_closest_coin, movement_action_reward, bomb_start_action_reward, bomb_action_reward


# Hyper parameters -- DO modify
GAMMA = 0.7
BATCH_SIZE = 128
MAT_SIZE = COLS * ROWS
HIDDEN_SIZE = 64
STEP_SIZE = 10000
LEARNING_RATE = 0.3

# Events
IN_BOMB_RADIUS = "IN_BOMB_RADIUS"
BOMB_EVADED = "BOMB_EVADED"
NOT_MOVING = "NOT_MOVING"
ACTION_PENALTY = "ACTION_PENALTY"
GOOD_MOVEMENT_ACTION = "GOOD_MOVEMENT_ACTION"
BAD_MOVEMENT_ACTION = "BAD_MOVEMENT_ACTION"
GOOD_BOMB_ACTION = "GOOD_BOMB_ACTION"
BAD_BOMB_ACTION = "BAD_BOMB_ACTION"
GOOD_BOMB_ACTION_START = "GOOD_BOMB_ACTION_START"
BAD_BOMB_ACTION_START = "BAD_BOMB_ACTION_START"


def setup_training(self):
    """
    Initialize the agent for training.
    """
    self.logger.info("Setting up the training environment.")
    self.visited_tiles = []
    self.same_position = 0
    self.action_history = deque(maxlen = 10)

    # Initialize the policy network and the target network
    self.policy_net = DQN(MAT_SIZE,
                          len(ACTIONS),
                          HIDDEN_SIZE)
    self.target_net = DQN(MAT_SIZE,
                          len(ACTIONS),
                          HIDDEN_SIZE)

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

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step.
    Consult settings.py to see what events are tracked.
    You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from
        `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) \
                      {", ".join(map(repr, events))} \
                      in step {new_game_state["step"]}')

    check_conditions(self,
                     old_game_state,
                     new_game_state,
                     events,
                     self_action)

    self.visited_tiles.append(new_game_state["self"][3])
    self.memory.push(state_to_features(old_game_state),
                     self_action,
                     state_to_features(new_game_state),
                     reward_from_events(self,
                                        events))
    optimize_model(self,
                   self_action)


def check_conditions(self,
                     old_game_state: dict,
                     new_game_state: dict,
                     events: List[str],
                     self_action: str):
    """
    Check specific conditions and append corresponding events.
    """

    if len(new_game_state["bombs"]) != 0:
        if check_danger_zone(new_game_state,
                             BOMB_POWER):
            self.logger.debug("Event: IN_BOMB_RADIUS")
            events.append(IN_BOMB_RADIUS)

        self.logger.debug("Event: BOMB_EVADED")
        events.append(BOMB_EVADED)

    reward = find_closest_coin(self,
                               old_game_state,
                               new_game_state)
    agent_position = new_game_state["self"][3]
    is_good_movement = movement_action_reward(old_game_state,
                                              new_game_state,
                                              self_action)
    is_visited_tile = agent_position in self.visited_tiles

    if is_good_movement and not is_visited_tile and reward == 1:
        self.logger.debug("Event: GOOD_MOVEMENT_ACTION")
        events.append(GOOD_MOVEMENT_ACTION)
    
    if not is_good_movement and is_visited_tile and reward == 0:
        self.logger.debug("Event: BAD_MOVEMENT_ACTION")
        events.append(BAD_MOVEMENT_ACTION)

    if bomb_start_action_reward(old_game_state,
                                new_game_state):
        self.logger.debug("Event: BAD_BOMB_ACTION_START")
        events.append(BAD_BOMB_ACTION_START)
    else:
        self.logger.debug("Event: GOOD_BOMB_ACTION_START")
        events.append(GOOD_BOMB_ACTION_START)

    if bomb_action_reward(old_game_state,
                          new_game_state,
                          self_action):
        self.logger.debug("Event: GOOD_BOMB_ACTION")
        events.append(GOOD_BOMB_ACTION)
    else:
        self.logger.debug("Event: BAD_BOMB_ACTION")
        events.append(BAD_BOMB_ACTION)


def end_of_round(self,
                 last_game_state: dict,
                 last_action: str,
                 events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
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


def reward_from_events(self,
                       events: List[str]) -> int:
    """
    Calculate the total reward based on a dictionary of event rewards.

    Args:
        events (List[str]): List of events that occurred during the game.

    Returns:
        int: The total reward for the given events.
    """

    # Define a dictionary to map events to rewards
    event_rewards = {
        e.MOVED_LEFT: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_DOWN: 0.1,
        e.WAITED: -0.8,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: -0.05,
        e.BOMB_EXPLODED: 0.5,
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0.5,
        e.COIN_COLLECTED: 0.5,
        #e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -1,
        #e.GOT_KILLED: -1,
        #e.OPPONENT_ELIMINATED: 0.7,
        #e.SURVIVED_ROUND: 1,
        IN_BOMB_RADIUS: -0.5,
        BOMB_EVADED: 0.3,
        GOOD_MOVEMENT_ACTION: 0.3,
        BAD_MOVEMENT_ACTION: -0.4,
        GOOD_BOMB_ACTION: 0.5,
        BAD_BOMB_ACTION: -0.7,
        GOOD_BOMB_ACTION_START: 0.5,
        BAD_BOMB_ACTION_START: -1
    }

    # Calculate the total reward for the given events
    total_reward = sum(event_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {total_reward} for events {', '.join(events)}")
    total_reward -= 0.01

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
        dtype=torch.bool)
    non_final_next_states = torch.from_numpy(
        np.array([s for s in batch.next_state if s is not None], dtype=np.float32)
    )

    # Convert batched states to a PyTorch tensor
    state_batch = torch.from_numpy(np.array(batch.state,
                                            dtype=np.float32))

    # Create a one-hot encoded tensor for the selected actions in the batch
    action_indices = [ACTIONS.index(action) for action in batch.action]
    action_batch = torch.zeros(BATCH_SIZE,
                               len(ACTIONS),
                               dtype=torch.float32)
    action_batch[range(BATCH_SIZE), action_indices] = 1

    # Compute Q-values for the selected actions in the current policy network
    state_action_values = self.policy_net(
        state_batch).gather(1,torch.tensor(action_indices).unsqueeze(1))

    # Compute the maximum Q-values for non-final next states using the target network
    next_state_values = torch.zeros(BATCH_SIZE,
                                    dtype=torch.float32)
    next_state_values[non_final_mask] = self.target_net(
        non_final_next_states).max(1).values.detach()

    # Compute the expected Q-values (target) using the Bellman equation
    reward_batch = torch.tensor(batch.reward,
                                dtype=torch.float32)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the loss using the Huber loss (smooth L1 loss)
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model by backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.scheduler.step()
