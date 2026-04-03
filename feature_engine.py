import pandas as pd
import numpy as np


class FeatureEngine:

    def __init__(self):
        pass

    # -------------------------------------------------------
    # 1️⃣ Tagesaggregation auf CET-Basis
    # -------------------------------------------------------
    def build_daily_structure(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Erwartet:
        - 1h OHLCV Daten
        - UTC tz-aware Index
        """

        # --- CET Konvertierung ---
        df = df_1h.tz_convert("Europe/Berlin")

        # ------------------------------------
        # Tages OHLC Aggregation (CET)
        # ------------------------------------
        daily = df.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        daily = daily.dropna()

        # ------------------------------------
        # Basic Daily Features
        # ------------------------------------
        daily["daily_return"] = daily["close"].pct_change()
        daily["daily_range"] = (daily["high"] - daily["low"]) / daily["open"]

        # ------------------------------------
        # Trend Features (nur abgeschlossene Tage)
        # ------------------------------------
        daily["trend1"] = daily["close"].shift(1).pct_change(1)
        daily["trend2"] = daily["close"].shift(1).pct_change(2)
        daily["trend3"] = daily["close"].shift(1).pct_change(3)
        daily["trend4"] = daily["close"].shift(1).pct_change(4)

        # ------------------------------------
        # Weekday / Weekend
        # ------------------------------------
        daily["weekday"] = daily.index.weekday
        daily["is_weekend"] = daily["weekday"] >= 5
        daily["is_sunday"] = daily["weekday"] == 6
        daily["is_monday"] = daily["weekday"] == 0

        # ------------------------------------
        # 6h Session Ranges
        # ------------------------------------
        session_ranges = self._calculate_6h_ranges(df)

        daily = daily.merge(session_ranges, left_index=True, right_index=True, how="left")

        return daily

    # -------------------------------------------------------
    # 2️⃣ 6h Session Range Berechnung
    # -------------------------------------------------------
    def _calculate_6h_ranges(self, df_cet: pd.DataFrame) -> pd.DataFrame:
        """
        Erwartet CET 1h Daten (tz-aware Europe/Berlin).
        """

        df = df_cet.copy()

        df["date"] = df.index.normalize()  # behält Timezone!
        df["hour"] = df.index.hour

        sessions = {
            "range_00_06": (0, 6),
            "range_06_12": (6, 12),
            "range_12_18": (12, 18),
            "range_18_24": (18, 24),
        }

        result = []

        for date, group in df.groupby("date"):

            day_dict = {"date": date}  # date ist jetzt tz-aware

            day_open = group.iloc[0]["open"]

            for name, (start, end) in sessions.items():
                session = group[(group["hour"] >= start) & (group["hour"] < end)]

                if len(session) > 0:
                    high = session["high"].max()
                    low = session["low"].min()
                    day_dict[name] = (high - low) / day_open
                else:
                    day_dict[name] = np.nan

            result.append(day_dict)

        session_df = pd.DataFrame(result)
        session_df.set_index("date", inplace=True)

        return session_df