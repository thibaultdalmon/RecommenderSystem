import DQN

class Trainer:

    def __init__(self, interface):
        self.dqn = DQN()
        self.interface = interface

    def train():
        item_history = self.interface.item_history
        rating_history = self.interface.rating_history
        user_history = self.interface.user_history
        DQN.train(user_history, item_history, rating_history)

    def predict(user_id, item_id):
        DQN.predict(user_id, item_id)
