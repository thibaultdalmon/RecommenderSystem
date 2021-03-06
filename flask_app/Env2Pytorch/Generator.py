import torch
from torch.utils.data import Dataset
import numpy as np


class PosNegData:

    def __init__(self, pos_data, neg_data, weight):
        self.pos = pos_data
        self.neg = neg_data
        self.weight = weight


class Data:

    def __init__(self, user_id, item_id, metadata):
        self.user_id = torch.tensor(user_id)
        self.item_id = torch.tensor(item_id)
        self.metadata = torch.tensor(metadata)


class DataGenerator(Dataset):

    def __init__(self, state_history, reward_history, action_history):
        self.state_history = state_history
        self.reward_history = reward_history
        self.action_history = action_history
        self.data = []
        self._init_pos_neg()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _init_pos_neg(self):
        for i, r in enumerate(self.reward_history):
            if r > 0:
                user_id = self.state_history[i][0][0]
                action = self.action_history[i]
                pos_data = Data(user_id=user_id, item_id=self.state_history[i][action][1],
                                metadata=self.state_history[i][action][2:])
                for j, state in enumerate(self.state_history[i]):
                    item_id = state[1]
                    metadata = state[2:]
                    data = Data(user_id=user_id, item_id=item_id, metadata=metadata)
                    if j != action:
                        self.data.append(PosNegData(pos_data, data, 1e9))

    def add_data(self, state, action, reward):
        if reward > 0:
            user_id = state[0][0]
            pos_data = Data(user_id=user_id, item_id=state[action][1], metadata=state[action][2:])
            for j, my_state in enumerate(state):
                item_id = my_state[1]
                metadata = my_state[2:]
                data = Data(user_id=user_id, item_id=item_id, metadata=metadata)
                if j != action:
                    self.data.append(PosNegData(pos_data, data, 1e9))


def collate_data_pos_neg(list_of_data):
    raw_data = [data for data in list_of_data]
    user_id_pos = torch.stack([data.pos.user_id for data in list_of_data])
    item_id_pos = torch.stack([data.pos.item_id for data in list_of_data])
    metadata_pos = torch.stack([data.pos.metadata for data in list_of_data])
    user_id_neg = torch.stack([data.neg.user_id for data in list_of_data])
    item_id_neg = torch.stack([data.neg.item_id for data in list_of_data])
    metadata_neg = torch.stack([data.neg.metadata for data in list_of_data])
    return {'user_id_pos': user_id_pos, 'item_id_pos': item_id_pos, 'metadata_pos': metadata_pos, 'raw_data': raw_data,
            'user_id_neg': user_id_neg, 'item_id_neg': item_id_neg, 'metadata_neg': metadata_neg}


def collate_data(list_of_data):
    user_id = torch.stack([data.user_id for data in list_of_data])
    item_id = torch.stack([data.item_id for data in list_of_data])
    metadata = torch.stack([data.metadata for data in list_of_data])
    return {'user_id': user_id, 'item_id': item_id, 'metadata': metadata}
