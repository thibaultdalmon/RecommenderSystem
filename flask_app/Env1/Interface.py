import requests


class Interface:

    def __init__(self, args):
        self.base_url = 'http://{}'.format(args.ip_address_new_env)
        self.user_id = args.user_id
        self.url_reset = '{}/reset'.format(self.base_url)
        self.url_predict = '{}/predict'.format(self.base_url)

        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()

        self.item_history = data['item_history']
        self.rating_history = data['rating_history']
        self.user_history = data['user_history']
        self.variables_history = data['variables_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']

        self.next_user = data['next_user']
        self.next_item = data['next_item']
        self.next_variables = data['next_variables']

    def reset(self):

        r = requests.get(url=self.url_reset, params={'user_id': self.user_id})
        data = r.json()

        self.item_history = data['item_history']
        self.rating_history = data['rating_history']
        self.user_history = data['user_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']

        self.next_user = data['next_user']
        self.next_item = data['next_item']

    def request(self, predicted_score):
        params = {}
        params['user_id'] = self.user_id
        params['predicted_score'] = predicted_score

        r = requests.get(url=self.url_predict, params=params)
        result = r.json()

        next_user = result['next_user']
        next_item = result['next_item']
        rating = result['rating']

        return next_user, next_item, rating
