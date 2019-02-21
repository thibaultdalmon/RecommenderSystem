from keras.utils import Sequence
import numpy as np

from collections import defaultdict


class Data:

    def __init__(self, user_id, item_id, metadata):
        self.user_id = user_id
        self.item_id = item_id
        self.metadata = metadata


class DataGenerator(Sequence):

    def __init__(self, alpha, state_history, reward_history, action_history, batch_size=32, n_sample=1000):
        self.alpha = alpha
        self.state_history = state_history
        self.reward_history = reward_history
        self.action_history = action_history
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.nb_variables = len(self.state_history[0]) - 2
        self.positive_data = defaultdict(list)
        self.negative_data = defaultdict(list)
        self.nb_positive = 0
        self._init_pos_neg()
        self.data = []
        self._generate_data()

    def __len__(self):
        return int(self.n_sample / self.batch_size)

    def __getitem__(self, idx):
        user_p = np.empty((self.batch_size, 1))
        item_p = np.empty((self.batch_size, 1))
        metadata_p = np.empty((self.batch_size, self.nb_variables))
        user_n = np.empty((self.batch_size, 1))
        item_n = np.empty((self.batch_size, 1))
        metadata_n = np.empty((self.batch_size, self.nb_variables))

        for i in range(self.batch_size):
            user_p[i] = self.data[idx * self.batch_size + i][0]
            item_p[i] = self.data[idx * self.batch_size + i][1]
            metadata_p[i] = self.data[idx * self.batch_size + i][2]
            user_n[i] = self.data[idx * self.batch_size + i][3]
            item_n[i] = self.data[idx * self.batch_size + i][4]
            metadata_n[i] = self.data[idx * self.batch_size + i][5]

        alphas = self.alpha * np.ones((self.batch_size, 1))
        return self.data[idx], alphas

    def on_epoch_end(self):
        self._generate_data()
        np.random.shuffle(self.data)

    def _init_pos_neg(self):
        for i, r in enumerate(self.reward_history):
            user = self.state_history[i][0]
            item = self.state_history[i][1]
            metadata = self.state_history[i][2:]
            if r > 0:
                self.nb_positive += 1
                self.positive_data[user].append(Data(user_id=user, item_id=item, metadata=metadata))
            else:
                self.negative_data[user].append(Data(user_id=user, item_id=item, metadata=metadata))

    def _generate_data(self):
        self.data = []
        for user_p, pos_data in self.positive_data.items():
            neg_data = np.random.choice(self.negative_data[user_p])
            self.data.append((pos_data.user_id, pos_data.item_id, pos_data.metadata, neg_data.user_id, neg_data.item_id,
                             neg_data.metadata))
