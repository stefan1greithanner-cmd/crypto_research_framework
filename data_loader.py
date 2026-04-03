import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")
        
        df = pd.read_parquet(file_path)

        # --- Index Validation ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex.")

        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware (UTC expected).")

        if str(df.index.tz) != "UTC":
            raise ValueError("DatetimeIndex must be UTC.")

        # Sort & remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        return df

    def load_multiple_assets(self, asset_files: dict) -> dict:
        """
        Lädt mehrere Assets aus Parquet-Dateien.

        asset_files Beispiel:
        {
            "BTC": "btc_1m_all.parquet",
            "ETH": "eth_1m_all.parquet",
            "LINK": "link_1m_all.parquet"
        
        }
        """

        assets = {}

        for asset, filename in asset_files.items():

            print(f"Loading {asset} from {filename}")

            df = self.load_parquet(filename)

            assets[asset] = df

        return assets
    
    def resample_multiple_assets(self, assets: dict, timeframe: str) -> dict:

        resampled = {}

        for asset, df in assets.items():

            resampled[asset] = self.resample_ohlcv(df, timeframe)

        return resampled

    def filter_date_range(
        self,
        df: pd.DataFrame,
        start: str = None,
        end: str = None
    ) -> pd.DataFrame:
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }

        df_resampled = df.resample(timeframe).agg(ohlc_dict)
        df_resampled = df_resampled.dropna()

        return df_resampled

    def add_cet_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp_cet"] = df.index.tz_convert("Europe/Berlin")
        df["hour_cet"] = df["timestamp_cet"].dt.hour
        df["date_cet"] = df["timestamp_cet"].dt.date
        return df