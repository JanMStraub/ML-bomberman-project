import pickle
import torch
import torch.nn.functional as F
from torch import optim
from typing import List

import events as e
import numpy as np
from .callbacks import state_to_features, ACTIONS
from .q_network import DQN
from .replay_memory import ReplayMemory, Transition
from settings import COLS, ROWS

# Hyper parameters -- DO modify
GAMMA = 0.95
BATCH_SIZE = 128
MAT_SIZE = COLS * ROWS
HIDDEN_SIZE = 64

# Events
ALREADY_VISITED = "ALREADY_VISITED"


def setup_training(self):
    """
    Initialize the agent for training.
    """
    self.logger.info("Setting up the training environment.")
    self.visited_tiles = []

    # Initialize the policy network and the target network
    self.policy_net = DQN(MAT_SIZE, len(ACTIONS), HIDDEN_SIZE)
    self.target_net = DQN(MAT_SIZE, len(ACTIONS), HIDDEN_SIZE)

    # Load the initial weights of the target network
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    self.memory = ReplayMemory(10000)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.memory.push(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))
    optimize_model(self, self_action)
    
    check_conditions(self, new_game_state, events)


def check_conditions(self, new_game_state: dict, events: List[str]):
    """
    Check specific conditions and append corresponding events.
    """
    if new_game_state["self"][3] in self.visited_tiles:
        events.append(ALREADY_VISITED)
        
    self.visited_tiles.append(new_game_state["self"][3])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
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
    self.memory.push(state_to_features(last_game_state), last_action, None, reward_from_events(self, events))
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str]) -> int:
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
        e.WAITED: -0.5,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.5,
        e.BOMB_EXPLODED: 0.5,
        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 10,
        e.SURVIVED_ROUND: 10,
        ALREADY_VISITED: -0.2,
    }

    # Calculate the total reward for the given events
    total_reward = sum(event_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {total_reward} for events {', '.join(events)}")
    total_reward -= 0.1

    return total_reward


def optimize_model(self, action):
    if len(self.memory) < BATCH_SIZE:
        return

    # Sample a batch of transitions from the replay memory
    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Create a mask to identify non-final next states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)

    # Filter out non-final next states and convert them to a tensor
    # Combine the list of NumPy arrays into a single NumPy array
    non_final_next_states = np.array([s for s in batch.next_state if s is not None], dtype=np.float32)

    # Convert the NumPy array to a PyTorch tensor
    non_final_next_states = torch.from_numpy(non_final_next_states)

    # Convert the batch data into tensors
    # Combine the list of NumPy arrays into a single NumPy array
    state_batch = np.array(batch.state, dtype=np.float32)

    # Convert the NumPy array to a PyTorch tensor
    state_batch = torch.from_numpy(state_batch)
    action_batch = torch.zeros((BATCH_SIZE, len(ACTIONS)), dtype=torch.int64)

    # Use a list comprehension to get the action indices for each item in batch.action
    action_indices = [ACTIONS.index(action) for action in batch.action]

    # Use advanced indexing to set the corresponding elements in action_batch to 1
    action_batch[range(BATCH_SIZE), action_indices] = 1

    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

    # Compute Q-values for the current state-action pairs
    state_action_values = self.policy_net(state_batch).gather(1, action_batch.argmax(dim=1, keepdim=True))

    # Compute the expected Q-values for the next states
    next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float32)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1).values

    # Compute the expected Q-values using the Bellman equation
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()