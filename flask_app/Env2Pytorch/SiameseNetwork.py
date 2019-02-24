from torch import nn
from torch.nn import Embedding, Linear, Bilinear, BatchNorm1d, ReLU, Dropout


class SiameseNetwork(nn.Module):

    def __init__(self, interface):
        super(SiameseNetwork, self).__init__()

        user_embedding_dim = 10
        item_embedding_dim = 10
        user_meta_dim = 15
        item_meta_dim = 15
        meta_meta_dim = 30
        dense_1_dim = 32
        dense_2_dim = 15
        out_dim = 1

        self.embedding_user = Embedding(num_embeddings=interface.nb_users, embedding_dim=user_embedding_dim)
        self.embedding_item = Embedding(num_embeddings=interface.nb_items, embedding_dim=item_embedding_dim)
        self.concat_user_meta = Bilinear(in1_features=user_embedding_dim, in2_features=interface.nb_variables, out_features=user_meta_dim)
        self.concat_item_meta = Bilinear(in1_features=item_embedding_dim, in2_features=interface.nb_variables, out_features=item_meta_dim)
        self.concat_meta_meta = Bilinear(in1_features=user_meta_dim, in2_features=item_meta_dim, out_features=meta_meta_dim)
        self.batch_norm_0 = BatchNorm1d(num_features=meta_meta_dim)
        self.dropout_0 = Dropout(0.5)
        self.dense_1 = Linear(in_features=meta_meta_dim, out_features=dense_1_dim)
        self.relu_1 = ReLU()
        self.dropout_1 = Dropout(0.5)
        self.batch_norm_1 = BatchNorm1d(num_features=dense_1_dim)
        self.dense_2 = Linear(in_features=dense_1_dim, out_features=dense_2_dim)
        self.relu_2 = ReLU()
        self.dropout_2 = Dropout(0.5)
        self.batch_norm_2 = BatchNorm1d(num_features=dense_2_dim)
        self.dense_3 = Linear(in_features=dense_2_dim, out_features=out_dim)

    def forward(self, user_id, item_id, metadata):
        user_embedded = self.embedding_user(user_id).squeeze(dim=1)
        item_embedded = self.embedding_item(item_id).squeeze(dim=1)
        user_and_meta = self.concat_user_meta(user_embedded, metadata)
        item_and_meta = self.concat_item_meta(item_embedded, metadata)
        meta_and_meta = self.concat_meta_meta(user_and_meta, item_and_meta)
        output = self.batch_norm_0(meta_and_meta)
        output = self.dropout_0(output)
        output = self.dense_1(output)
        output = self.relu_1(output)
        output = self.batch_norm_1(output)
        output = self.dropout_1(output)
        output = self.dense_2(output)
        output = self.relu_2(output)
        output = self.batch_norm_2(output)
        output = self.dropout_2(output)
        output = self.dense_3(output)
        return output
