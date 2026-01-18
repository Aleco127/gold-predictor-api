"""
Test script for Capital.com API connection.
Run this to verify your credentials are working.
"""

import sys
sys.path.insert(0, ".")

from src.data.capital_connector import CapitalConnector
from loguru import logger

# Your Capital.com credentials
API_KEY = "yK7S4GB0zu1RmJcU"
PASSWORD = "pixqep-piRgo1-vimbod"
EMAIL = "alejandrocorral27@gmail.com"
USE_DEMO = False  # Live account (no demo account exists)


def main():
    logger.info("Testing Capital.com API connection...")

    # Create connector
    connector = CapitalConnector(
        api_key=API_KEY,
        password=PASSWORD,
        identifier=EMAIL,
        demo=USE_DEMO,
        symbol="GOLD",
    )

    try:
        # Test connection
        logger.info(f"Connecting to Capital.com {'demo' if USE_DEMO else 'live'}...")
        connector.connect()
        logger.success("Connection successful!")

        # Test getting current price
        logger.info("Fetching current GOLD price...")
        price = connector.get_current_price("GOLD")
        if price:
            logger.success(f"Current GOLD price:")
            logger.info(f"  Bid: ${price['bid']:.2f}")
            logger.info(f"  Ask: ${price['ask']:.2f}")
            logger.info(f"  Spread: ${price['spread']:.2f}")
            logger.info(f"  Market Status: {price.get('status', 'N/A')}")

        # Test getting symbol info
        logger.info("Fetching GOLD market info...")
        info = connector.get_symbol_info("GOLD")
        if info:
            logger.success(f"Market info:")
            logger.info(f"  Name: {info.get('name', 'N/A')}")
            logger.info(f"  Type: {info.get('type', 'N/A')}")
            logger.info(f"  Currency: {info.get('currency', 'N/A')}")
            logger.info(f"  Min Quantity: {info.get('min_quantity', 'N/A')}")

        # Test getting historical data
        logger.info("Fetching 100 bars of M5 data...")
        df = connector.get_ohlcv(symbol="GOLD", timeframe="M5", bars=100)
        if not df.empty:
            logger.success(f"Received {len(df)} bars of data")
            logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
            logger.info(f"  Columns: {list(df.columns)}")
        else:
            logger.warning("No historical data received")

        # Search for gold-related markets
        logger.info("Searching for gold markets...")
        markets = connector.search_markets("gold", limit=5)
        if markets:
            logger.success(f"Found {len(markets)} gold-related markets:")
            for m in markets:
                logger.info(f"  - {m['symbol']}: {m['name']} ({m['type']})")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        logger.info("Disconnecting...")
        connector.disconnect()

    logger.success("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
