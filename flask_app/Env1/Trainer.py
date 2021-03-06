from Env1.DQN import DQN


class Trainer:

    def __init__(self, interface):
        self.interface = interface
        self.dqn = DQN(interface)
        self.train()

    def train(self):
        item_history = self.interface.item_history
        rating_history = self.interface.rating_history
        user_history = self.interface.user_history
        variables_history = self.interface.variables_history
        self.dqn.train(user_history, item_history, variables_history, rating_history)

    def reset(self):
        self.dqn.reset()
        self.train()

    def predict(self, user_id, item_id, metadata):
        return self.dqn.predict(user_id, item_id, metadata).item()
