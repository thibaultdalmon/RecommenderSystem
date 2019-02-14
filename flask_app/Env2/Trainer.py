from Env2.DQN import DQN


class Trainer:

    def __init__(self, interface):
        self.interface = interface
        self.dqn = DQN(interface)
        self.train()

    def train(self):
        state_history = self.interface.state_history
        reward_history = self.interface.reward_history
        action_history = self.interface.action_history
        self.dqn.train(state_history, reward_history, action_history)

    def reset(self):
        self.dqn.reset()
        self.train()

    def predict(self, user_id, item_id, metadata):
        return self.dqn.predict(user_id, item_id, metadata).item()
