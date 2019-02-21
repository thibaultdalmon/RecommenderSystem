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
        generator_train = DataGenerator(0.1, state_history, rewards_history, action_history)
        generator_val = DataGenerator(0.1, state_history, rewards_history, action_history)
        self.dqn.train(generator_train, generator_val)

    def reset(self):
        self.dqn.reset()
        self.train()

    def predict(self, user_id, item_id, metadata):
        return self.dqn.predict(user_id, item_id, metadata).item()
