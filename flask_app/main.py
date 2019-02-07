import argparse
from flask import Flask, request, jsonify
import ./OldEnv as old

app = Flask(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--user_id', type=str, default='R3EIFXNYY6XMBXBR01BK')
parser.add_argument('--ip_address_old_env', type=str, default='52.47.62.31')
parser.add_argument('--ip_address_new_env', type=str, default='35.180.46.68')

args = parser.parse_args()

interface = old.Interface(args)
trainer = Trainer()

@app.route("/reset")
def reset():
    interface.reset()
    trainer.train()

@app.route("/request", methods=['GET', 'POST'])
def predict():
    user_id = request.args.get('user_id')
    item_id = request.args.get('item_id')
    predicted_score = trainer.predict(user_id, item_id)
    d = {'predicted_score' : predicted_score}
    return jsonify(d)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
