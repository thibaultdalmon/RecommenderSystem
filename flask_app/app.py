from flask import Flask, request, jsonify
from OldEnv.Interface import Interface
from OldEnv.Trainer import Trainer

app = Flask(__name__)


class Argument:
    pass


args = Argument
args.user_id = 'R3EIFXNYY6XMBXBR01BK'
args.ip_address_old_env = '52.47.62.31'
args.ip_address_new_env = '35.180.46.68'

interface = Interface(args)
trainer = Trainer(args)


@app.route("/reset")
def reset():
    interface.reset()
    trainer.train()


@app.route("/request", methods=['GET', 'POST'])
def predict():
    user_id = request.args.get('user_id')
    item_id = request.args.get('item_id')
    predicted_score = trainer.predict(user_id, item_id)
    d = {'predicted_score': predicted_score}
    return jsonify(d)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
