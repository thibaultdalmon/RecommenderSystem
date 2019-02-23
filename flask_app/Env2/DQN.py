from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, merge
from keras.layers import Dropout, Subtract, Add, Maximum, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

def custom_loss(y_true, y_pred):
    import keras.backend as K
    y = y_pred[0]-y_pred[1]
    return K.mean(K.maximum(y+10,0))

class DQN:

    def __init__(self, interface):
        # For each sample we input the integer identifiers
        # of a single user and a single item
        user_id_input_p = Input(shape=(1,), name='user_p')
        item_id_input_p = Input(shape=(1,), name='item_p')
        metadata_input_p = Input(shape=(interface.nb_variables,), name='metadata_p')
        user_id_input_n = Input(shape=(1,), name='user_n')
        item_id_input_n = Input(shape=(1,), name='item_n')
        metadata_input_n = Input(shape=(interface.nb_variables,), name='metadata_n')

        batch_metadata_p = metadata_input_p
        batch_metadata_n = metadata_input_n

        embedding_size = 10
        user_embedding = Embedding(output_dim=embedding_size,
                                   input_dim=interface.nb_users,
                                   input_length=1, name='user_embedding')
        user_embedding_p = user_embedding(user_id_input_p)
        user_embedding_n = user_embedding(user_id_input_n)

        embedding_size = 5
        embedding_item = Embedding(output_dim=embedding_size,
                                   input_dim=interface.nb_items,
                                   input_length=1, name='item_embedding')
        item_embedding_p = embedding_item(item_id_input_p)
        item_embedding_n = embedding_item(item_id_input_n)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs_p = Flatten()(user_embedding_p)
        user_vecs_n = Flatten()(user_embedding_n)
        item_vecs_p = Flatten()(item_embedding_p)
        item_vecs_n = Flatten()(item_embedding_n)
        # metadata_vecs = Flatten()(metadata_input)
        conc = Concatenate(axis=1)
        conc_p = conc([user_vecs_p, item_vecs_p, batch_metadata_p])
        conc_n = conc([user_vecs_n, item_vecs_n, batch_metadata_n])

        batch_2 = BatchNormalization()
        batch_2_p = batch_2(conc_p)
        batch_2_n = batch_2(conc_n)

        dense_1 = Dense(8, activation='relu')
        dense_1_p = dense_1(batch_2_p)
        dense_1_n = dense_1(batch_2_n)

        batch_3 = BatchNormalization()
        batch_3_p = batch_3(dense_1_p)
        batch_3_n = batch_3(dense_1_n)

        dropout_1 = Dropout(0.2)
        dropout_1_p = dropout_1(batch_3_p)
        dropout_1_n = dropout_1(batch_3_n)

        dense_2 = Dense(8, activation='relu')
        dense_2_p = dense_2(dropout_1_p)
        dense_2_n = dense_2(dropout_1_n)

        batch_4 = BatchNormalization()
        batch_4_p = batch_4(dense_2_p)
        batch_4_n = batch_4(dense_2_n)

        dropout_2 = Dropout(0.2)
        dropout_2_p = dropout_2(batch_4_p)
        dropout_2_n = dropout_2(batch_4_n)

        dense_3 = Dense(1)
        dense_3_n = dense_3(dropout_2_n)
        dense_3_p = dense_3(dropout_2_p)

        self.model = Model(inputs=[user_id_input_p, item_id_input_p,
                                   metadata_input_p, user_id_input_n, item_id_input_n,
                                   metadata_input_n], outputs=[dense_3_n, dense_3_p])

        self.model.compile(optimizer=Adam(1e-03), loss=custom_loss)
        self.model.save('Env2/Models/initial_weight.h5')

    def reset(self):
        self.model = load_model('Env2/Models/initial_weight.h5',
                custom_objects={'custom_loss':custom_loss})

    def train(self, generator_train, generator_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit_generator(generator=generator_train,
                                 epochs=50,
                                 validation_data=generator_val,
                                 shuffle=True,  # callbacks=[early_stopping],
                                 use_multiprocessing=True, workers=2,
                                 max_queue_size=32)
        print('Fin training')
        return

    def predict(self, generator):
        return self.model.predict_generator(generator, steps=1)
