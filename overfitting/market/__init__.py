"""
Market data from exchanges.

    load        download (once) and return candles and funding for a backtest
    stored      what the local store holds
    downloads   the store's log: every download, and files added, changed or deleted by hand
    EXCHANGES   the supported exchanges

    from overfitting.market import load

    data = load(["BTC", "ETH"], "1h", start="2023-01-01", exchange="binance")
"""
from overfitting.errors import ExchangeError
from overfitting.market.exchanges import EXCHANGES, TIMEFRAMES
from overfitting.market.loader import downloads, load, parse_symbol, stored

__all__ = ["load", "stored", "downloads", "parse_symbol", "EXCHANGES", "TIMEFRAMES", "ExchangeError"]
