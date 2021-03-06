from Env2.DQN import DQN
from Env2.Generator import DataGenerator

import numpy as np


class Trainer:

    def __init__(self, interface):
        self.interface = interface
        self.dqn = DQN(interface)

    def train(self):
        state_history = self.interface.state_history
        rewards_history = self.interface.rewards_history
        action_history = self.interface.action_history
        generator_train = DataGenerator(state_history, rewards_history, action_history)
        generator_val = DataGenerator(state_history, rewards_history, action_history)
        self.dqn.train(generator_train, generator_val)

    def reset(self):
        self.dqn.reset()
        prediction = self.predict()
        res1 = np.concatenate([prediction[0], prediction[1]], axis=1)
        self.train()
        prediction = self.predict()
        res2 = np.concatenate([prediction[0], prediction[1]], axis=1)
        self.train()
        print("Prediction")
        prediction = self.predict()
        res3 = np.concatenate([prediction[0], prediction[1]], axis=1)
        print(np.concatenate([res1, res2, res3], axis=1))

    def predict(self):
        state_history = self.interface.state_history
        rewards_history = self.interface.rewards_history
        action_history = self.interface.action_history
        generator_pred = DataGenerator(state_history, rewards_history, action_history, mode="prediction")
        return self.dqn.predict(generator_pred)
