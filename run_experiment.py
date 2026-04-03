from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine


# --------------------------------------------------
# 1️⃣ Daten laden
# --------------------------------------------------

loader = DataLoader("D:/research_framework/data")

df = loader.load_parquet("btc_1m_all.parquet")

df = loader.filter_date_range(df, start="2022-01-01")

df_1h = loader.resample_ohlcv(df, "1h")

df_1h = loader.add_cet_columns(df_1h)


# --------------------------------------------------
# 2️⃣ Feature Engine
# --------------------------------------------------

engine = FeatureEngine()

daily_structure = engine.build_daily_structure(df_1h)


# --------------------------------------------------
# 3️⃣ Output
# --------------------------------------------------

print(daily_structure.head())
print(daily_structure.tail())
print(daily_structure.describe())
from core.data_loader import DataLoader

loader = DataLoader("D:/research_framework/data")

df = loader.load_parquet("btc_1m_all.parquet")

df = loader.filter_date_range(df, start="2022-01-01")

df_1h = loader.resample_ohlcv(df, "1h")

df_1h = loader.add_cet_columns(df_1h)

print(df_1h.head())
print(df_1h.tail())

from analysis.descriptive_stats import DescriptiveStats

stats = DescriptiveStats(daily_structure)

print("\n===== BASIC STATS =====")
print(stats.basic_stats())

print("\n===== WEEKDAY RETURNS =====")
print(stats.weekday_returns())

print("\n===== WEEKEND EFFECT =====")
print(stats.weekend_effect())

print("\n===== TREND CONDITIONAL =====")
print(stats.trend_conditional())

print("\n===== SESSION VOLATILITY =====")
print(stats.session_volatility())

from analysis.regime_analysis import RegimeAnalysis

regime = RegimeAnalysis(daily_structure)

print("\n===== QUIET MORNING → US EXPANSION =====")
print(regime.quiet_morning_expansion())

print("\n===== VOLATILITY CLUSTERING =====")
print(regime.volatility_clustering())

print("\n===== BIG MOVE FOLLOW THROUGH =====")
print(regime.big_move_follow_through())

print(daily_structure["daily_return"].describe())

big_up = daily_structure[daily_structure["daily_return"] > 0.04]

print(big_up[["daily_return"]].head())

from analysis.regime_switching import RegimeSwitching

regimes = RegimeSwitching(daily_structure)

print("\n===== VOLATILITY REGIMES =====")
print(regimes.volatility_regimes())

print("\n===== TREND PERSISTENCE =====")
print(regimes.trend_persistence())

print("\n===== DIRECTION PERSISTENCE =====")
print(regimes.direction_persistence())