from keras.layers import Input, Embedding, Flatten, Dot
from keras.models import Model
from keras.callbacks import EarlyStopping

class DQN():

    def __init__(self, interface):
        # For each sample we input the integer identifiers
        # of a single user and a single item
        user_id_input = Input(shape=[1],name='user')
        item_id_input = Input(shape=[1], name='item')

        embedding_size = 30
        user_embedding = Embedding(output_dim=embedding_size, input_dim=n_user + 1,
                                   input_length=1, name='user_embedding')(user_id_input)

        item_embedding = Embedding(output_dim=embedding_size, input_dim=n_item + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        y = Dot(axes=1)([user_vecs, item_vecs])

        self.dqn = Model(inputs=[user_id_input, item_id_input], outputs=y)
        self.dqn.compile(optimizer=..., loss=...)

    def train(user_id_train, item_id_train, rating_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, callbacks=[early_stopping])

    def.predict(user_id, item_id):
        model.predict([user_id, item_id])
