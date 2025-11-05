import os

from binance.client import Client as BinanceClient
from binance.enums import *
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

binance_client = BinanceClient(api_key, secret_key)

account = binance_client.get_account()

for asset in account.get('balances'):
    if float(asset.get('free')) > 0:
        print(
            f'{asset.get("asset")}: {asset.get("free")} - Locked: {asset.get("locked")}'
        )
