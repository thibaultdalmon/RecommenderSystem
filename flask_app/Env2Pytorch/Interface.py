import requests
from tqdm import tqdm


class Interface:

    def __init__(self, args):
        self.base_url = 'http://{}'.format(args.ip_address_env_2)
        self.user_id = args.user_id
        self.url_reset = '{}/reset'.format(self.base_url)
        self.url_predict = '{}/predict'.format(self.base_url)

        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()
        self.state_history = data['state_history']
        self.rewards_history = data['rewards_history']
        self.action_history = data['action_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']
        self.nb_variables = len(self.state_history[0][0]) - 2

        self.next_state = data['next_state']

    def reset(self):
        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()

        self.state_history = data['state_history']
        self.rewards_history = data['rewards_history']
        self.action_history = data['action_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']

        self.next_state = data['next_state']

    def predict(self, recommended_item):
        r = requests.get(url=self.url_predict, params={'user_id': self.user_id, 'recommended_item': recommended_item})
        data = r.json()

        self.state_history.append(data['state'])
        self.rewards_history.append(data['reward'])
        self.action_history.append(recommended_item)

        self.next_state = data['state']
        return data['state'], data['reward']
