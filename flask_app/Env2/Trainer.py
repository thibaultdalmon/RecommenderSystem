from Env2.DQN import DQN
from Env2.Generator import DataGenerator


class Trainer:

    def __init__(self, interface):
        self.interface = interface
        self.dqn = DQN(interface)
        self.train()

    def train(self):
        state_history = self.interface.state_history
        rewards_history = self.interface.rewards_history
        action_history = self.interface.action_history
        generator_train = DataGenerator(state_history, rewards_history, action_history)
        generator_val = DataGenerator(state_history, rewards_history, action_history)
        self.dqn.train(generator_train, generator_val)

    def reset(self):
        self.dqn.reset()
        self.train()
        print("Prediction")
        prediction = self.predict()
        print(prediction)

    def predict(self):
        state_history = self.interface.state_history
        rewards_history = self.interface.rewards_history
        action_history = self.interface.action_history
        generator_pred = DataGenerator(state_history, rewards_history,
        action_history)
        return self.dqn.predict(generator_pred).item()
