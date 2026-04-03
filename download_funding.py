import requests
import pandas as pd
import time
from datetime import datetime, timedelta

BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BASE_PATH = r"D:\research_framework\data\derivatives\funding"


def download_funding(symbol):

    start = datetime(2019, 1, 1)
    end = datetime.utcnow()

    window = timedelta(days=60)

    all_data = []

    while start < end:

        end_window = start + window

        params = {
            "symbol": symbol,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end_window.timestamp() * 1000),
            "limit": 1000
        }

        r = requests.get(BASE_URL, params=params)

        if r.status_code != 200:
            print("API error:", r.text)
            break

        data = r.json()

        print(symbol, start.date(), "→", len(data), "records")

        all_data.extend(data)

        start = end_window

        time.sleep(0.2)

    df = pd.DataFrame(all_data)

    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)

    df = df[["timestamp", "funding_rate"]]

    df = df.drop_duplicates("timestamp")

    return df


def save_funding(symbol, filename):

    df = download_funding(symbol)

    path = f"{BASE_PATH}/{filename}"

    df.to_parquet(path)

    print("Saved:", path)


if __name__ == "__main__":

    save_funding("BTCUSDT", "btc_funding.parquet")
    save_funding("ETHUSDT", "eth_funding.parquet")
    save_funding("LINKUSDT", "link_funding.parquet")