import pandas as pd

BASE_PATH = r"D:\research_framework\data"


def load_price(asset):

    path = f"{BASE_PATH}/{asset}_1m_all.parquet"

    df = pd.read_parquet(path)

    # Sicherstellen dass Index datetime ist
    df.index = pd.to_datetime(df.index)

    # Nur benötigte Spalten behalten (Performance)
    df = df[["open", "high", "low", "close", "volume"]]

    return df

def resample_1h(df):

    df_1h = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    return df_1h

def load_funding(asset):

    path = f"{BASE_PATH}/derivatives/funding/{asset}_funding.parquet"

    funding = pd.read_parquet(path)

    funding["timestamp"] = pd.to_datetime(funding["timestamp"])
    funding = funding.set_index("timestamp")
    funding.index = pd.to_datetime(funding.index, utc=True)
    funding = funding.resample("1h").ffill()

    return funding

def merge_data(price, funding):

    df = price.join(funding, how="left")

    df["funding_rate"] = df["funding_rate"].ffill()

    return df

def compute_features(df):

    lookback = 24 * 30

    df["funding_mean"] = df["funding_rate"].rolling(lookback).mean()
    df["funding_std"] = df["funding_rate"].rolling(lookback).std()

    df["funding_z"] = (
        df["funding_rate"] - df["funding_mean"]
    ) / df["funding_std"]

    return df

def compute_forward_returns(df):

    df["ret_6h"] = df["close"].pct_change(6).shift(-6)
    df["ret_12h"] = df["close"].pct_change(12).shift(-12)
    df["ret_24h"] = df["close"].pct_change(24).shift(-24)
    df["ret_48h"] = df["close"].pct_change(48).shift(-48)
    df["ret_72h"] = df["close"].pct_change(72).shift(-72)

    return df

def bucket_analysis(df):
    df = df.dropna()
    
    # Entferne NaNs
    df = df.dropna(subset=["funding_z"])

    # Buckets erstellen
    df["bucket"] = pd.qcut(df["funding_z"], 5)

    print("\nBucket Counts")
    print(df["bucket"].value_counts().sort_index())

    horizons = ["ret_6h","ret_12h","ret_24h","ret_48h","ret_72h"]

    print("\nBucket Mean Returns")

    for h in horizons:

        result = df.groupby("bucket")[h].mean()

        print(f"\n{h}")
        print(result)

    # Extreme funding analyse
    extreme = df[abs(df["funding_z"]) > 2]

    print("\nExtreme Funding Count:", len(extreme))

    print("\nExtreme Funding Returns")

    for h in horizons:

        print(f"{h}: {extreme[h].mean()}")

        long_extreme = df[df["funding_z"] < -2]
    short_extreme = df[df["funding_z"] > 2]

    print("\nLong Extreme Count:", len(long_extreme))
    print("Short Extreme Count:", len(short_extreme))

    print("\nLong Extreme Returns")

    for h in horizons:
        print(f"{h}: {long_extreme[h].mean()}")

    print("\nShort Extreme Returns")

    for h in horizons:
        print(f"{h}: {short_extreme[h].mean()}")

def run(asset):

    print("Running", asset)

    price = load_price(asset)
    price = resample_1h(price)

    funding = load_funding(asset)

    df = merge_data(price, funding)

    df = compute_features(df)
    df = compute_forward_returns(df)

    bucket_analysis(df)

for asset in ["btc", "eth", "link"]:
    run(asset)
    