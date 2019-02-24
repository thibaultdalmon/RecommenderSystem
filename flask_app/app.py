from flask import Flask, request, jsonify
from Env0.Interface import Interface as Interface0
from Env0.Trainer import Trainer as Trainer0
# from Env1.Interface import Interface as Interface1
# from Env1.Trainer import Trainer as Trainer1
# from Env2Pytorch.Interface import Interface as Interface2
# from Env2Pytorch.Trainer import Trainer as Trainer2

import numpy as np

app = Flask(__name__)


class Argument:
    pass


args = Argument
args.user_id = 'R3EIFXNYY6XMBXBR01BK'
args.ip_address_env_0 = '52.47.62.31'
args.ip_address_env_1 = '35.180.254.42'
args.ip_address_env_2 = '35.180.178.243'

args.use_env = 0

interface = None
trainer = None
if args.use_env == 0:
    interface = Interface0(args)
    trainer = Trainer0(interface)
elif args.use_env == 1:
    interface = Interface1(args)
    trainer = Trainer1(interface)
elif args.use_env == 2:
    interface = Interface2(args)
    trainer = Trainer2(interface)
else:
    raise Exception('Unknown environment: {}'.format(args.use_env))

@app.route("/reset", methods=['GET', 'POST'])
def reset():
    interface.reset()
    trainer.reset()

@app.route("/train", methods=['GET', 'POST'])
def train():
    data = request.get_json(force=True)
    interface.item_history = data['item_history']
    interface.rating_history = data['rating_history']
    interface.user_history = data['user_history']

    interface.nb_items = data['nb_items']
    interface.nb_users = data['nb_users']

    interface.next_user = data['next_user']
    interface.next_item = data['next_item']
    trainer.reset()


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    user = np.array([[float(request.args.get('user'))]])
    item = np.array([[float(request.args.get('item'))]])
    predicted_score = trainer.predict(user, item)
    d = {'predicted_score': predicted_score}
    return jsonify(d)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
