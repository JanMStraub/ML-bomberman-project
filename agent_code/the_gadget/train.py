import pickle
import torch
import torch.nn.functional as F
from torch import optim
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS
from .model import DQN
from .replay_memory import ReplayMemory

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
GAMMA = 0.99

# Events
VICTORY = "VICTORY"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up the training setup.")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    self.policy_net = DQN(578, 6)
    self.target_net = DQN(578, 6)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    self.memory = ReplayMemory(10000)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(VICTORY)

    self.memory.push(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))
    optimize_model(self, old_game_state, self_action, new_game_state, events)
    
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.transitions.append(Transition(state_to_features(last_game_state), [last_action], None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: .1,
        e.MOVED_RIGHT: .1,
        e.MOVED_UP: .1,
        e.MOVED_DOWN: .1,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: -.5,
        e.BOMB_EXPLODED: .5,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100,
        e.OPPONENT_ELIMINATED: 10,
        e.SURVIVED_ROUND: 10
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    reward_sum -= .1
    return reward_sum

# TODO
def optimize_model(self, old_state, action, new_state, events):
    if old_state is None or new_state is None:
            return
        
    old_game_state = torch.tensor(state_to_features(old_state), dtype=torch.float32)
    new_game_state = torch.tensor(state_to_features(new_state), dtype=torch.float32)
    reward = reward_from_events(self, events)

    action_idx = ACTIONS.index(action)
    action_mask = torch.zeros(len(ACTIONS), dtype=torch.float32)
    action_mask[action_idx] = 1.0

    state_action_value = self.policy_net(old_game_state).squeeze()[action_idx]
    next_state_action_value = self.target_net(new_game_state).max().unsqueeze(0)

    expected_state_action_value = (next_state_action_value * GAMMA) + reward
    loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()