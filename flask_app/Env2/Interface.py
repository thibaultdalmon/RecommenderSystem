import requests


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

    def request(self, predicted_score):
        params = {}
        params['user_id'] = self.user_id
        params['predicted_score'] = predicted_score

        r = requests.get(url=self.url_predict, params=params)
        result = r.json()

        next_user = result['next_user']
        next_item = result['next_item']
        next_variables = result['next_variables']
        rating = result['rating']

        return next_user, next_item, next_variables, rating
