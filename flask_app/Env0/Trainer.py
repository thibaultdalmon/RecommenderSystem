from Env0.DQN import DQN


class Trainer:

    def __init__(self, interface):
        self.interface = interface
        self.dqn = DQN(interface)

    def train(self):
        item_history = self.interface.item_history
        rating_history = self.interface.rating_history
        user_history = self.interface.user_history
        self.dqn.train(user_history, item_history, rating_history)

    def reset(self):
        self.dqn.reset()
        self.train()

    def predict(self, user_id, item_id):
        return self.dqn.predict(user_id, item_id).item()
