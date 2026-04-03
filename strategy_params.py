PARAMETER_SPACE = {

    "volatility_trend_strategy": {

        "volatility_threshold": [60, 70, 80, 90],

        "trend_length": [1, 2, 3, 4],

        "session_filter": [
            None,
            "EU",
            "US"
        ]

    }

}
PARAMETER_SPACE["momentum_5d_strategy"] = {
    "lookback": [3,5,7],
    "holding": [5,10,15],
    "threshold": [0.01,0.02,0.03]
}
PARAMETER_SPACE["volatility_breakout_strategy"] = {
    "lookback": [5,10,20],
    "compression_quantile": [0.1,0.2,0.3],
    "holding": [5,10,15]
}