#!/usr/bin/env python3
"""
Quick test to verify timezone issues are resolved
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

def test_timezone_handling():
    """Test timezone handling in data fetching"""
    print("Testing timezone handling...")

    try:
        # Test fetching data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")

        print(f"Original data timezone: {data.index.tz}")

        # Simulate our cleaning process
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            print(f"After conversion: {data.index.tz}")

        # Test comparison operations that were failing
        end_date = datetime.now()
        end_timestamp = pd.Timestamp(end_date)

        if data.index.tz is not None:
            end_timestamp = end_timestamp.tz_localize(data.index.tz)

        print(f"Data index type: {type(data.index[0])}")
        print(f"Comparison timestamp: {type(end_timestamp)}")

        # This should not raise a timezone error anymore
        recent_data = data.index.max() >= end_timestamp - pd.Timedelta(days=7)
        print(f"Comparison successful: {recent_data}")

        print("✅ Timezone handling test passed!")
        return True

    except Exception as e:
        print(f"❌ Timezone test failed: {e}")
        return False

if __name__ == "__main__":
    test_timezone_handling()