import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--user_id', type=str, default='R3EIFXNYY6XMBXBR01BK')
parser.add_argument('--ip_address_old_env', type=str, default='52.47.62.31')
parser.add_argument('--ip_address_new_env', type=str, default='35.180.46.68')

args = parser.parse_args()