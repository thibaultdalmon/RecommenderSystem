from Env2Pytorch.SiameseNetwork import SiameseNetwork
from Env2Pytorch.Generator import DataGenerator, collate_data_pos_neg, Data, collate_data

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self, interface, nb_epoch=10, learning_rate=3e-4, validation_split=0.2, batch_size=32, margin=10):
        self.interface = interface
        self.network = SiameseNetwork(interface)
        self.dataset = DataGenerator(interface.state_history, interface.rewards_history, interface.action_history)
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.validation_split = validation_split

        self.loss = MarginRankingLoss(margin=margin, reduction='sum')

        self.optimizer = Adam(self.network.parameters(), lr=learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=0.3, patience=5, threshold=1e-3, verbose=True)

    def reset(self):
        self.train()
        while True:
            self.online()

    def train(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_data_pos_neg, drop_last=True)
        self.network.train()
        for i in range(self.nb_epoch):
            cumloss = 0
            for inputs in data_loader:
                self.optimizer.zero_grad()
                output_pos = self.network(inputs['user_id_pos'], inputs['item_id_pos'], inputs['metadata_pos'])
                output_neg = self.network(inputs['user_id_neg'], inputs['item_id_neg'], inputs['metadata_neg'])
                loss = self.loss(output_pos, output_neg, torch.ones(output_pos.shape))
                cumloss += loss.item()
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
        print(reward)
        self.train()



