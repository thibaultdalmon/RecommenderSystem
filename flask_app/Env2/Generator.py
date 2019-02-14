from keras.utils import Sequence


class Data:

    def __init__(self, user_id, item_id, metadata):
        self.user_id = user_id
        self.item_id = item_id
        self.metadata = metadata


class DataGenerator(Sequence):

    def __init__(self, state_history, reward_history, action_history, batch_size=32, n_sample=1000):
        self.state_history = state_history
        self.reward_history = reward_history
        self.action_history = action_history
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.data = self._generate_data()

    def __len__(self):
        return int(self.n_sample / self.batch_size)

    def __getitem__(self, idx):
        return self.data[idx], 0

    def on_epoch_end(self):
        pass

    def _generate_data(self):
        return
