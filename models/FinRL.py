import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import talib
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

# Enhanced FinancialEnvironment
class FinancialEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001, max_drawdown=0.2):
        super(FinancialEnvironment, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.max_drawdown = max_drawdown
        self.reset()
        
        # Expanded action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_value = self.balance
        self.max_value = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.balance / self.initial_balance,
            self.shares_held,
            self.current_price / self.df['Close'].max(),
            self.df['Close'][self.current_step] / self.df['Close'].max(),
            self.df['Volume'][self.current_step] / self.df['Volume'].max(),
            self.df['RSI'][self.current_step] / 100,
            self.df['MACD'][self.current_step] / self.df['MACD'].max(),
            self.df['ATR'][self.current_step] / self.df['ATR'].max(),
            self.df['BB_upper'][self.current_step] / self.df['Close'].max(),
            self.df['BB_lower'][self.current_step] / self.df['Close'].max(),
            self.df['OBV'][self.current_step] / self.df['OBV'].max(),
            self.df['ADX'][self.current_step] / 100,
            self.df['CCI'][self.current_step] / 200 + 0.5,  # Normalize to 0-1
            self.df['MOM'][self.current_step] / self.df['MOM'].max(),
            (self.total_value - self.initial_balance) / self.initial_balance  # Relative profit/loss
        ])
        return frame

    def step(self, action):
        self.current_price = self.df['Close'][self.current_step]
        current_value = self.balance + self.shares_held * self.current_price

        # Execute trade
        if action > 0:  # Buy
            shares_to_buy = action * self.balance / self.current_price
            cost = shares_to_buy * self.current_price * (1 + self.transaction_fee_percent)
            self.balance -= cost
            self.shares_held += shares_to_buy
        elif action < 0:  # Sell
            shares_to_sell = -action * self.shares_held
            income = shares_to_sell * self.current_price * (1 - self.transaction_fee_percent)
            self.balance += income
            self.shares_held -= shares_to_sell

        self.current_step += 1
        new_value = self.balance + self.shares_held * self.current_price
        self.total_value = new_value
        self.max_value = max(self.max_value, new_value)

        # Calculate reward with risk management
        reward = (new_value - current_value) / current_value
        drawdown = (self.max_value - new_value) / self.max_value
        if drawdown > self.max_drawdown:
            reward -= 1  # Penalize for exceeding max drawdown

        done = self.current_step >= len(self.df) - 1 or drawdown > self.max_drawdown

        return self._next_observation(), reward, done, {}

# Enhanced technical indicators calculation
def calculate_indicators(df):
    df['RSI'] = talib.RSI(df['Close'])
    df['MACD'], _, _ = talib.MACD(df['Close'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    df['MOM'] = talib.MOM(df['Close'])
    return df.dropna()

# Custom Neural Network
class FinancialCNN(tf.keras.Model):
    def __init__(self, input_shape, action_dim):
        super(FinancialCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape)
        self.lstm = tf.keras.layers.LSTM(128)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(action_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.lstm(x)
        x = self.dense1(x)
        return self.dense2(x)

# Ensemble Model
class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, observation):
        predictions = [model.predict(observation) for model in self.models]
        return np.mean(predictions, axis=0)

# Adaptive Learning Rate
class AdaptiveLearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience=5):
        super(AdaptiveLearningRateCallback, self).__init__()
        self.factor = factor
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = self.model.optimizer.lr * self.factor
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch}: reducing learning rate to {new_lr}")
                self.wait = 0

# Curriculum Learning
def curriculum_learning(env, model, stages):
    for stage in stages:
        env.max_drawdown = stage['max_drawdown']
        env.transaction_fee_percent = stage['transaction_fee']
        model.learn(total_timesteps=stage['timesteps'])