import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class FinancialMarketEnvironment:
    def __init__(self, data, transaction_cost=0.001, max_position=1.0):
        self.data = data
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.balance = 10000  # Starting balance

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        return self._get_observation()

    def step(self, action):
        current_price = self.data[self.current_step]['close']
        next_price = self.data[self.current_step + 1]['close']
        
        # Interpret continuous action
        position_delta = action[0] * self.max_position
        take_profit = action[1]
        stop_loss = action[2]

        # Apply position change
        old_position = self.position
        self.position = np.clip(self.position + position_delta, -self.max_position, self.max_position)
        
        # Calculate transaction cost
        transaction_cost = abs(self.position - old_position) * current_price * self.transaction_cost
        self.balance -= transaction_cost

        # Calculate profit/loss
        if old_position != 0:
            profit = (current_price - self.entry_price) * old_position
            self.balance += profit

        # Check for stop loss or take profit
        if self.position != 0:
            if (self.position > 0 and current_price <= self.entry_price * (1 - stop_loss)) or \
               (self.position < 0 and current_price >= self.entry_price * (1 + stop_loss)):
                # Stop loss hit
                self.balance += self.position * (current_price - self.entry_price)
                self.position = 0
            elif (self.position > 0 and current_price >= self.entry_price * (1 + take_profit)) or \
                 (self.position < 0 and current_price <= self.entry_price * (1 - take_profit)):
                # Take profit hit
                self.balance += self.position * (current_price - self.entry_price)
                self.position = 0

        # Update entry price if position changed
        if self.position != old_position:
            self.entry_price = current_price

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self._get_observation()
        reward = self._calculate_reward(next_price)
        
        return next_state, reward, done

    def _get_observation(self):
        # Return relevant market data for the current step
        return np.array([
            self.data[self.current_step]['open'],
            self.data[self.current_step]['high'],
            self.data[self.current_step]['low'],
            self.data[self.current_step]['close'],
            self.data[self.current_step]['volume'],
            self.position,
            self.balance
        ])

    def _calculate_reward(self, next_price):
        # Calculate reward based on balance change and risk-adjusted return
        balance_change = self.balance - 10000  # Change from starting balance
        position_value = self.position * next_price
        sharpe_ratio = balance_change / (np.std([self.balance, 10000]) + 1e-9)  # Add small value to avoid division by zero
        return balance_change + sharpe_ratio * 100  # Weight Sharpe ratio more heavily

class TradingNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(TradingNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        policy = torch.tanh(self.policy_head(shared_output))  # Output in range [-1, 1]
        value = self.value_head(shared_output)
        return policy, value

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if action not in self.children:
                self.children[action] = MCTSNode(None, self)
                self.children[action].prior = prob

    def select_child(self):
        c_puct = 1.0
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            uct_score = child.get_value() + c_puct * child.prior * (np.sqrt(self.visit_count) / (1 + child.visit_count))
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(value)

class MCTS:
    def __init__(self, network, num_simulations=100):
        self.network = network
        self.num_simulations = num_simulations

    def search(self, state, env):
        root = MCTSNode(state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.children:
                action, node = node.select_child()
                search_path.append(node)

            # Expansion and evaluation
            parent = search_path[-2]
            state = env._get_observation()
            policy, value = self.network(torch.FloatTensor(state).unsqueeze(0))
            policy = policy.detach().numpy().flatten()
            value = value.item()

            # If the node was not expanded before, expand it
            if not node.children:
                node.expand(policy)

            # Backup
            for node in reversed(search_path):
                node.backup(value)

        # Select action based on visit counts
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        action = np.argmax(visit_counts)
        return action

class TradingAlphaZero:
    def __init__(self, input_dim, action_dim, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = TradingNetwork(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.mcts = MCTS(self.network)
        self.action_dim = action_dim