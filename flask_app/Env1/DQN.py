from keras.layers import (Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout)
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping


class DQN:

    def __init__(self, interface):
        # For each sample we input the integer identifiers
        # of a single user and a single item
        user_id_input = Input(shape=(1,), name='user')
        item_id_input = Input(shape=(1,), name='item')
        metadata_input = Input(shape=(interface.nb_variables,), name='metadata')

        embedding_size = 30
        user_embedding = Embedding(output_dim=embedding_size, input_dim=interface.nb_users + 1,
                                   input_length=1, name='user_embedding')(user_id_input)

        item_embedding = Embedding(output_dim=embedding_size, input_dim=interface.nb_items + 1,
                                   input_length=1, name='item_embedding')(item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)
        # metadata_vecs = Flatten()(metadata_input)

        conc = Concatenate(axis=1)([user_vecs, item_vecs, metadata_input])

        dense_1 = Dense(256, activation='relu')(conc)

        dropout_1 = Dropout(0.5)(dense_1)

        dense_2 = Dense(128, activation='relu')(dropout_1)

        dropout_2 = Dropout(0.5)(dense_2)

        y = Dense(1)(dropout_2)

        self.model = Model(inputs=[user_id_input, item_id_input,
            metadata_input], outputs=y)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.save('Env1/Models/initial_weight.h5')

    def reset(self):
        self.model = load_model('Env1/Models/initial_weight.h5')

    def train(self, user_id_train, item_id_train, metadata, rating_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit([user_id_train, item_id_train, metadata], rating_train,
                       batch_size=64, epochs=1, validation_split=0.1,
                       shuffle=True, callbacks=[early_stopping])

    def predict(self, user_id, item_id, metadata):
        return self.model.predict([user_id, item_id, metadata])
