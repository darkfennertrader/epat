import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from talib import BBANDS

ticker = "MSFT"
start = datetime(2015, 1, 2)
end = datetime(2020, 4, 30)


def download_daily_data(ticker, start, end):
    """
    The function downloads daily market data to a pandas DataFrame
    using the 'yfinance' API between the dates specified.
    """
    data = yf.download(ticker, start, end)

    return data


def compute_indicators(data):
    """
    The function creates additional columns to an OHLC pandas DataFrame
    required to backtest the "Bollinger Bands" trading strategy.
    """
    # Columns created to check condition (ii) & (iii):
    data["previous Adj Close"] = data["Adj Close"].shift(1)
    data["up_prev_day"] = data["UpperBB"].shift(1)
    data["low_prev_day"] = data["LowerBB"].shift(1)

    # condition (ii):
    data["signal"] = np.where(
        (data["previous Adj Close"] > data["low_prev_day"])
        & (data["Adj Close"] < data["LowerBB"]),
        1,
        0,
    )

    # condition (iii):
    data["signal"] = np.where(
        (data["previous Adj Close"] < data["up_prev_day"])
        & (data["Adj Close"] > data["UpperBB"]),
        -1,
        data["signal"],
    )

    # condition (iv):
    data["position"] = data["signal"].replace(to_replace=0, method="ffill")

    data["position"] = data["position"].shift(1)

    return data


def backtest_strategy(data):
    """
    The function creates additional columns to the pandas DataFrame for checking conditions
    to backtest the "Bollinger Bands" trading strategy.
    It then computes the strategy returns.
    IMPORTANT: To be run ONLY after the function compute_indicators.
    """

    data["bnh_returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    # data['bbands_returns'] = data['bnh_returns'] * data['position'].shift(1)
    data["bbands_returns"] = data["bnh_returns"] * data["position"]

    logreturns = np.round(data["bnh_returns"].cumsum()[-1], 2)
    print(
        f'"Buy and hold" Strategy logreturns: {logreturns} converted to simple returns: {(np.exp(logreturns) - 1):.2f}'
    )

    # logreturns = np.round((data['buy_and_sell_returns']).cumsum()[-1], 2)

    logreturns = np.round((data["bbands_returns"]).cumsum()[-1], 2)
    print(
        f'"Bollinger Bands" Strategy logreturns: {logreturns} converted to simple returns: {(np.exp(logreturns) - 1):.2f}'
    )
    # data[['bnh_returns', 'bbands_returns']].cumsum().plot(grid=True, figsize=(12, 8));

    print(
        "\nNOTE: the total simple Returns of the strategy is the same calculated with pyfolio."
    )

    # we need these columns for using pyfolio to show the equivalence of the two approaches. It accepts only simple returns
    data["simple_returns"] = data["Adj Close"].pct_change()
    # data['bbands_pyfolio'] = data['simple_returns'] * data['position'].shift(1)
    data["bbands_pyfolio"] = data["simple_returns"] * data["position"]
    data["bbands_pyfolio"] = 1 + data["bbands_pyfolio"]

    print(f"BBANDS simple returns: {data['bbands_pyfolio'].cumprod()[-1] -1}")

    return data


if __name__ == "__main__":
    pass

    # df = download_daily_data(ticker, start, end)
    # print(df.tail(3))
    # print(df.shape)

    # dfbb = df.copy()
    # # we don't need the mid band and use '_' to esclude the column
    # dfbb["UpperBB"], _, dfbb["LowerBB"] = BBANDS(
    #     dfbb["Adj Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    # )

    # dfbb = compute_indicators(dfbb)
    # dfbb = backtest_strategy(dfbb)
