import requests


class Interface:

    def __init__(self, args):
        self.base_url = 'http://{}'.format(args.ip_address_new_env)
        self.user_id = args.user_id
        self.url_reset = '{}/reset'.format(self.base_url)
        self.url_predict = '{}/predict'.format(self.base_url)

        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()

        self.state_history = data['state_history']
        self.reward_history = data['reward_history']
        self.action_history = data['action_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']
        self.nb_variables = len(self.action_history[0])-2

        self.next_state = data['next_state']

    def reset(self):

        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()

        self.state_history = data['state_history']
        self.reward_history = data['reward_history']
        self.action_history = data['action_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']
        self.nb_variables = len(self.action_history[0])-2

        self.next_state = data['next_state']

    def predict(self, recommended_item):
        params = {}
        params['user_id'] = self.user_id
        params['recommended_item'] = recommended_item

        r = requests.get(url=self.url_predict, params=params)
        result = r.json()

        reward = result['reward']
        state = result['state']

        return reward, state
