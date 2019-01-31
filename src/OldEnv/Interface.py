import requests


class Interface:

    def __init__(self, args):
        base_url = 'http://{}'.format(args.ip_address_old_env)
        url_reset = '{}/reset'.format(base_url)
        url_predict = '{}/predict'.format(base_url)

        r = requests.get(url=url_reset, params={'user_id': args.user_id})
        data = r.json()

        self.item_history = data['item_history']
        self.rating_history = data['rating_history']
        self.user_history = data['user_history']

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']

        self.next_user = data['next_user']
        self.next_item = data['next_item']
