from Env2Pytorch.SiameseNetwork import SiameseNetwork
from Env2Pytorch.Generator import DataGenerator, collate_data_pos_neg, Data, collate_data

import torch
from torch.optim import Adam
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader, WeightedRandomSampler


class Trainer:

    def __init__(self, interface, learning_rate=3e-4, batch_size=32, margin=10, num_samples=100, user_embedding_dim=10,
                 item_embedding_dim=10, user_meta_dim=15, item_meta_dim=15, meta_meta_dim=30, dense_1_dim=32,
                 dense_2_dim=15, dropout=0.5):
        self.interface = interface

        self.margin = margin
        self.learning_rate = learning_rate

        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.user_meta_dim = user_meta_dim
        self.item_meta_dim = item_meta_dim
        self.meta_meta_dim = meta_meta_dim
        self.dense_1_dim = dense_1_dim
        self.dense_2_dim = dense_2_dim
        self.dropout = dropout
        self.network = SiameseNetwork(interface, user_embedding_dim=self.user_embedding_dim,
                                      item_embedding_dim=item_embedding_dim, user_meta_dim=user_meta_dim,
                                      item_meta_dim=item_meta_dim, meta_meta_dim=meta_meta_dim, dense_1_dim=dense_1_dim,
                                      dense_2_dim=dense_2_dim, dropout=dropout)
        self.dataset = DataGenerator(interface.state_history, interface.rewards_history, interface.action_history)
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.loss = MarginRankingLoss(margin=margin, reduction='none')
        self.optimizer = Adam(self.network.parameters(), lr=learning_rate)

    def reset(self, n):
        self.network = SiameseNetwork(self.interface, user_embedding_dim=self.user_embedding_dim,
                                      item_embedding_dim=self.item_embedding_dim, user_meta_dim=self.user_meta_dim,
                                      item_meta_dim=self.item_meta_dim, meta_meta_dim=self.meta_meta_dim,
                                      dense_1_dim=self.dense_1_dim, dense_2_dim=self.dense_2_dim, dropout=self.dropout)
        self.dataset = DataGenerator(self.interface.state_history, self.interface.rewards_history,
                                     self.interface.action_history)
        self.loss = MarginRankingLoss(margin=self.margin, reduction='none')
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate)
        self.train(n)

    def train(self, n=1):
        for _ in range(n):
            weights = [data.weight for data in self.dataset]
            sampler = WeightedRandomSampler(weights=weights, num_samples=self.num_samples, replacement=True)
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                                     collate_fn=collate_data_pos_neg, drop_last=True)
            self.network.train()
            for inputs in data_loader:
                self.optimizer.zero_grad()
                output_pos = self.network(inputs['user_id_pos'], inputs['item_id_pos'], inputs['metadata_pos'])
                output_neg = self.network(inputs['user_id_neg'], inputs['item_id_neg'], inputs['metadata_neg'])
                loss = self.loss(output_pos, output_neg, torch.ones(output_pos.shape))
                for j, data in enumerate(inputs['raw_data']):
                    data.weight = loss[j][0].item()
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

    def online(self, n=1):
        self.network.eval()
        l = []
        my_state = self.interface.next_state
        for m in self.interface.next_state:
            data = Data(m[0], m[1], m[2:])
            l.append(data)
        inputs = collate_data(l)
        output = self.network(inputs['user_id'], inputs['item_id'], inputs['metadata']).squeeze()
        recommended_item = output.argmax().item()
        state, reward = self.interface.predict(recommended_item)
        self.dataset.add_data(my_state, recommended_item, reward)
        self.train(n=n)
        return reward
