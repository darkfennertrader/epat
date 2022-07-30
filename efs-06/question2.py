from turtle import color
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

filepath1 = "./Q2-Data/Nifty ETF.xlsx"
filepath2 = "./Q2-Data/Junior ETF.xlsx"
filepath3 = "./Q2-Data/Gold ETF.xlsx"


def read_data():
    # reading the file "Nifty ETF.xlsx"
    df1 = pd.read_excel(
        filepath1,
        skiprows=2,
        names=["Date", "Nifty"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )

    # reading the file "Junior ETF.xlsx"
    df2 = pd.read_excel(
        filepath2,
        skiprows=3,
        names=["Date", "Junior"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )

    # reading the file "Gold ETF.xlsx"
    df3 = pd.read_excel(
        filepath3,
        skiprows=4,
        names=["Date", "Gold"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )

    df = pd.concat([df1, df2, df3], axis=1)
    # dropping missing value from dataframe
    df.dropna(inplace=True)
    return df


def asset_allocation(
    df, starting_fund, date_start, date_end, df_previous=pd.DataFrame()
):
    date_start_str = datetime.strftime(date_start, "%d/%m/%Y")
    date_end_str = datetime.strftime(date_end, "%d/%m/%Y")
    print(f"\nPERIOD:")
    print(f"start date: {date_start_str}")
    print(f"end date: {date_end_str}")
    # print(f"shape before slicing: {df.shape}")
    df = df[date_start:date_end].copy()
    # print(f"shape after slicing: {df.shape}")

    # if first period normalize data
    if date_start == datetime(2016, 1, 1):
        print("DATAFRAME SETTINGS")
        for asset in ("Nifty", "Junior", "Gold"):
            df[asset + "_norm_ret"] = df[asset] / df.iloc[0][asset]

        for asset, allocation in zip(("Nifty", "Junior", "Gold"), [0.5, 0.2, 0.3]):
            df[asset + "_allocation"] = df[asset + "_norm_ret"] * allocation

        for asset in ("Nifty", "Junior", "Gold"):
            df[asset + "_pos"] = df[asset + "_allocation"] * starting_fund

        # df.drop(
        #     columns=[
        #         "Nifty",
        #         "Junior",
        #         "Gold",
        #         "Nifty_norm_ret",
        #         "Junior_norm_ret",
        #         "Gold_norm_ret",
        #         "Nifty_allocation",
        #         "Junior_allocation",
        #         "Gold_allocation",
        #     ],
        #     inplace=True,
        # )
        # df["Tot_pos"] = df.sum(axis=1)
        df["Tot_pos"] = df["Nifty_pos"] + df["Junior_pos"] + df["Gold_pos"]

        df["daily_ret"] = df["Tot_pos"].pct_change(1)
        # print("\ndataframe after adding the columns:")
        # print(df.head(3))
        # print(df.tail(3))
        df_post = df
        # print(f"\nshape after adding the columns: {df.shape}")
        return df_post

    else:
        print("\nPORTFOLIO REBALANCING")
        # print(df_previous.shape)
        # Portfolio re-balancing on the last working day of each quarter
        df_previous.loc[date_start, "Nifty_pos":"daily_ret"] = (
            df_previous["Tot_pos"].values[-1] * 0.5,
            df_previous["Tot_pos"].values[-1] * 0.2,
            df_previous["Tot_pos"].values[-1] * 0.3,
            df_previous["Tot_pos"].values[-1],
            df_previous["daily_ret"].values[-1],
        )

        # concat new dataframe with previous rebalanced one
        df_merged = pd.concat([df_previous, df.iloc[1:]])
        # print(df_previous.tail())
        # print(df_previous.shape)
        # print(df_merged.iloc[59:63])
        # print(f"before: {df_merged.shape}")
        df_merged = df_merged[date_start:date_end]
        # print(f"after: {df_merged.shape}")
        for asset in ("Nifty", "Junior", "Gold"):
            df_merged[asset + "_norm_ret"] = df_merged[asset] / df_merged.iloc[0][asset]

        for asset, allocation in zip(("Nifty", "Junior", "Gold"), [0.5, 0.2, 0.3]):
            df_merged[asset + "_allocation"] = (
                df_merged[asset + "_norm_ret"] * allocation
            )

        starting_fund = df_previous["Tot_pos"].values[-1]
        for asset in ("Nifty", "Junior", "Gold"):
            df_merged[asset + "_pos"] = df_merged[asset + "_allocation"] * starting_fund

        df_merged["Tot_pos"] = (
            df_merged["Nifty_pos"] + df_merged["Junior_pos"] + df_merged["Gold_pos"]
        )
        # print("\nDF_POST:")
        df_post = pd.concat([df_previous, df_merged[1:]])
        df_post["daily_ret"] = df_post["Tot_pos"].pct_change(1)
        # print(df_post.head(3))
        # print(df_post.tail(3))
        # print(df_post[57:66])
        # print(df_post.shape)

        # emit metrics at the end of the year
        if date_end in [datetime(2016, 12, 30), datetime(2017, 12, 29)]:
            year_str = datetime.strftime(date_end, "%Y")
            if date_end == datetime(2016, 12, 30):
                df_metric = df_post.loc["2016-01-01":"2016-12-30"]
            else:
                df_metric = df_post.loc["2017-01-01":"2017-12-30"]

            cagr = df_metric["Tot_pos"].values[-1] / df_metric["Tot_pos"].values[0] - 1
            sharp_ratio = (
                df_metric["daily_ret"].mean() / df_metric["daily_ret"].std()
            ) * (252**0.5)
            # fig, axs = plt.subplots(2)
            # fig.tight_layout()
            # df_metric[["Tot_pos"]].plot(
            #     ax=axs[0],
            #     color="black",
            #     grid=True,
            #     figsize=(20, 15),
            # )
            # df_metric[["Nifty_pos", "Junior_pos", "Gold_pos"]].plot(
            #     ax=axs[1],
            #     grid=True,
            #     figsize=(20, 15),
            # )

            plt.show()
            print(f"\nAnnualized Returns (CAGR) for {year_str}: {(cagr*100):.3f}%")
            # SHARPE ratio:
            print(f"Sharpe Ratio (SR) for {year_str}: {sharp_ratio:.3f}")

    return df_post


df = read_data()
eobq = pd.date_range(
    start="2016-01-01",
    end="2017-12-30",
    freq="BQ",
).to_list()

# print(eobq)

df_previous = pd.DataFrame()
starting_date = datetime(2016, 1, 1)
starting_fund = 100

for idx_list in range(len(eobq)):
    end_date = eobq[idx_list]
    df_output = asset_allocation(
        df, starting_fund, starting_date, end_date, df_previous
    )
    df_previous = df_output.copy()
    starting_date = end_date
