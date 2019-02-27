from flask import Flask, request, jsonify
# from Env0.Interface import Interface as Interface0
# from Env0.Trainer import Trainer as Trainer0
# from Env1.Interface import Interface as Interface1
# from Env1.Trainer import Trainer as Trainer1
from Env2Pytorch.Interface import Interface as Interface
from Env2Pytorch.Trainer import Trainer as Trainer

import numpy as np
import pandas as pd
from tqdm import tqdm

app = Flask(__name__)


class Argument:
    pass


# args = Argument
# args.user_id = 'R3EIFXNYY6XMBXBR01BK'
# args.ip_address_env_0 = '52.47.62.31'
# args.ip_address_env_1 = '35.180.254.42'
# args.ip_address_env_2 = '35.180.178.243'
#
# args.use_env = 2
#
# interface = None
# trainer = None
# if args.use_env == 0:
#     interface = Interface0(args)
#     trainer = Trainer0(interface)
# elif args.use_env == 1:
#     interface = Interface1(args)
#     trainer = Trainer1(interface)
# elif args.use_env == 2:
#     interface = Interface2(args)
#     trainer = Trainer2(interface)
# else:
#     raise Exception('Unknown environment: {}'.format(args.use_env))
#
# @app.route("/reset", methods=['GET', 'POST'])
# def reset():
#     interface.reset()
#     trainer.reset()
#
# @app.route("/train", methods=['GET', 'POST'])
# def train():
#     data = request.get_json(force=True)
#     interface.item_history = data['item_history']
#     interface.rating_history = data['rating_history']
#     interface.user_history = data['user_history']
#
#     interface.nb_items = data['nb_items']
#     interface.nb_users = data['nb_users']
#
#     interface.next_user = data['next_user']
#     interface.next_item = data['next_item']
#     trainer.reset()


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    user = np.array([[float(request.args.get('user'))]])
    item = np.array([[float(request.args.get('item'))]])
    predicted_score = trainer.predict(user, item)
    d = {'predicted_score': predicted_score}
    return jsonify(d)


if __name__ == '__main__':
    args = Argument()

    args.user_id = 'R3EIFXNYY6XMBXBR01BK'
    args.ip_address_env_2 = '35.180.178.243'

    parameters = pd.read_csv('../parameters.csv')

    n_heurisitc = parameters.shape[0]
    nb_iter = 100
    nb_episode = 100
    res_rewards = np.zeros((n_heurisitc, nb_iter, nb_episode))

    for idx, row in enumerate(parameters.itertuples()):
        print('{} / {}'.format(idx + 1, n_heurisitc))
        interface = Interface(args)
        trainer = Trainer(interface, learning_rate=row.lr, batch_size=row.batch_size, margin=row.margin,
                          num_samples=row.n_sample, user_embedding_dim=row.user_embedding_dim,
                          item_embedding_dim=row.item_embedding_dim, user_meta_dim=row.user_meta_dim,
                          item_meta_dim=row.item_meta_dim, meta_meta_dim=row.meta_meta_dim,
                          dense_1_dim=row.dense1, dense_2_dim=row.dense2, dropout=row.dropout)
        for i in tqdm(range(nb_iter)):
            interface.reset()
            trainer.reset(n=row.n_epoch_before)

            for j in range(nb_episode):
                res_rewards[idx, i, j] = trainer.online(n=row.n_epoch_online)
        np.savetxt('heuristic_{}.txt'.format(idx), res_rewards[idx])

