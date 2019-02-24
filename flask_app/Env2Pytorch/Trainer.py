from Env2Pytorch.SiameseNetwork import SiameseNetwork
from Env2Pytorch.Generator import DataGenerator, collate_data_pos_neg, Data, collate_data

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader, WeightedRandomSampler


class Trainer:

    def __init__(self, interface, learning_rate=3e-4, validation_split=0.2, batch_size=32, margin=10, min_weight=1,
                 num_samples=100):
        self.interface = interface
        self.network = SiameseNetwork(interface)
        self.dataset = DataGenerator(interface.state_history, interface.rewards_history, interface.action_history)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.min_weight = min_weight
        self.num_samples = num_samples

        self.loss = MarginRankingLoss(margin=margin, reduction='none')

        self.optimizer = Adam(self.network.parameters(), lr=learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.3, patience=5, threshold=1e-3, verbose=True)

    def reset(self):
        self.train()
        while True:
            self.online()

    def train(self):
        weights = [data.weight for data in self.dataset]
        sampler = WeightedRandomSampler(weights=weights, num_samples=self.num_samples, replacement=True)
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                                 collate_fn=collate_data_pos_neg, drop_last=True)
        self.network.train()
        cumloss = 0
        for inputs in data_loader:
            self.optimizer.zero_grad()
            output_pos = self.network(inputs['user_id_pos'], inputs['item_id_pos'], inputs['metadata_pos'])
            output_neg = self.network(inputs['user_id_neg'], inputs['item_id_neg'], inputs['metadata_neg'])
            loss = self.loss(output_pos, output_neg, torch.ones(output_pos.shape))
            for j, data in enumerate(inputs['raw_data']):
                data.weight = loss[j][0].item()
            cumloss += loss.sum().item()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
        print(cumloss / len(self.dataset))

    def online(self):
        self.network.eval()
        l = []
        my_state = self.interface.next_state
        for m in self.interface.next_state:
            data = Data(m[0], m[1], m[2:])
            l.append(data)
        input = collate_data(l)
        output = self.network(input['user_id'], input['item_id'], input['metadata']).squeeze()
        recommended_item = output.argmax().item()
        state, reward = self.interface.predict(recommended_item)
        self.dataset.add_data(my_state, recommended_item, reward)
        self.train()
