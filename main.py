import logging
import math
import os
import time

import pandas as pd
from binance.client import Client as BinanceClient
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()


api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

binance_client = BinanceClient(api_key, secret_key)

symbol = 'BTCBRL'
op_asset = 'BTC'
candle_interval = BinanceClient.KLINE_INTERVAL_1HOUR
quantity = 0.00001
actual_position = False

# symbol_info = binance_client.get_symbol_info(op_code)
# lot_size_filter = next(
#     f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'
# )
# min_qty = float(lot_size_filter['minQty'])
# max_qty = float(lot_size_filter['maxQty'])
# step_size = float(lot_size_filter['stepSize'])

# logger.info(f'Lot size filter: {lot_size_filter}')
# logger.info(f'Min quantity: {min_qty}')
# logger.info(f'Max quantity: {max_qty}')
# logger.info(f'Step size: {step_size}')


class BinanceBase(BaseModel):
    """Base class for Binance API"""

    client: BinanceClient
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GetData(BinanceBase):
    """Class for getting data from the Binance API."""

    symbol: str
    interval: str


class TradeStrategy(BinanceBase):
    """Class for trading configuration that inherits from the base class."""

    data: pd.DataFrame
    symbol: str
    asset: str
    quantity: float
    position: bool


def truncate(value: float, decimals: int = 3) -> float:
    """Truncate a float to a specific number of decimal places.

    Args:
        value: The float value to truncate
        decimals: Number of decimal places (default: 3)

    Returns:
        Truncated float value
    """
    factor = 10**decimals
    return math.floor(value * factor) / factor


def get_data(config: GetData) -> pd.DataFrame:
    candles = config.client.get_klines(
        symbol=config.symbol, interval=config.interval, limit=1000
    )
    prices = pd.DataFrame(candles)
    prices.columns = [
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_asset_volume',
        'number_of_trades',
        'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume',
        'ignore',
    ]

    prices = prices[['timestamp', 'open', 'high', 'low', 'close']]
    prices['timestamp'] = (
        pd.to_datetime(prices['timestamp'], unit='ms')
        .dt.tz_localize('UTC')
        .dt.tz_convert('America/Sao_Paulo')
    )

    return prices


def trade_strategy(config: TradeStrategy) -> bool:
    config.data['fast_average'] = config.data['close'].rolling(window=7).mean()
    config.data['slow_average'] = (
        config.data['close'].rolling(window=40).mean()
    )

    last_fast_average = config.data['fast_average'].iloc[-1]
    last_slow_average = config.data['slow_average'].iloc[-1]

    logger.info(
        f'Last fast average: {last_fast_average} | Last slow average: {last_slow_average}'
    )

    account = config.client.get_account()

    current_quantity = 0.0

    for account_asset in account['balances']:
        if account_asset['asset'] == config.asset:
            current_quantity = float(account_asset['free'])

    if last_fast_average > last_slow_average:
        if not config.position:
            order = config.client.create_order(
                symbol=config.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=config.quantity,
            )
            logger.info(
                f'Bought {config.quantity} {config.asset} at {order.get("price", "market")}'
            )
            config.position = True
    elif last_fast_average < last_slow_average:
        if config.position:
            order = config.client.create_order(
                symbol=config.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=truncate(current_quantity, 5),
            )
            logger.info(
                f'Sold {truncate(current_quantity, 5)} {config.asset} at {order.get("price", "market")}'
            )
            config.position = False

    return config.position


data_fetcher = GetData(
    client=binance_client, symbol=symbol, interval=candle_interval
)

strategy_config = TradeStrategy(
    client=binance_client,
    data=pd.DataFrame(),
    symbol=symbol,
    asset=op_asset,
    quantity=quantity,
    position=actual_position,
)

while True:
    refresh_data = get_data(data_fetcher)
    strategy_config.data = refresh_data
    strategy_config.position = actual_position
    actual_position = trade_strategy(strategy_config)
    time.sleep(15 * 60)
