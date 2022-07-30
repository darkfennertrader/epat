from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm


####################    QUESTION 1 functions    ################################


def read_data():
    """This function read and normalize all the data in a dataframe"""

    # reading the file "NIFTY-TotalReturnsIndex.csv"
    df1 = pd.read_csv(filepath1, index_col=0, parse_dates=True)
    # print(df1.head())
    # print(df1.tail())
    # print(df1.shape)

    # reading the file "Reliance Nifty ETF.xlsx"
    df2 = pd.read_excel(
        filepath2,
        skiprows=2,
        names=["Date", "Reliance"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )
    # print(df2.head())
    # print(df2.tail())
    # print(df2.shape)

    # reading the file "Kotak Nifty ETF.xlsx"
    df3 = pd.read_excel(
        filepath3,
        skiprows=3,
        names=["Date", "Kotak"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )
    # print(df3.head())
    # print(df3.tail())
    # print(df3.shape)

    # reading the file "HDFC Nifty ETF.xlsx"
    df4 = pd.read_excel(
        filepath4,
        skiprows=4,
        names=["Date", "HDFC"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )
    # print(df4.head())
    # print(df4.tail())
    # print(df4.shape)

    # reading the file "/UTI Nifty ETF.xlsx"
    df5 = pd.read_excel(
        filepath5,
        skiprows=2,
        names=["Date", "UTI"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )
    # print(df5.head())
    # print(df5.tail())
    # print(df5.shape)

    print()
    # concatenating  all the dataframes along their columns
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    # dropping missing value from dataframe
    df.dropna(inplace=True)
    print(df.head())
    # print(df.tail())
    print(f"Shape of original dataframe: {df.shape}")

    # print(df.isnull().sum())
    return df


def annualized_tracking_error(df):

    # split dataset by year
    df_2016 = df.loc["2016-01-01":"2016-12-31"].copy()
    df_2017 = df.loc["2017-01-01":"2017-12-31"].copy()

    # compute daily return calculation
    df_2016 = df_2016.pct_change().dropna().to_numpy()
    # split index returns from the rest for vectorized computation
    benchmark_2016 = df_2016[:, 0].reshape(-1, 1)
    etf_2016 = df_2016[:, 1:]
    print(f"\nShape of 2016 index array: {benchmark_2016.shape}")
    print(f"Shape of 2016 ETFs array: {etf_2016.shape}")

    # compute daily return calculation
    df_2017 = df_2017.pct_change().dropna().to_numpy()

    # split index returns from the rest for vectorized computation
    benchmark_2017 = df_2017[:, 0].reshape(-1, 1)
    etf_2017 = df_2017[:, 1:]
    print(f"\nShape of 2017 index array: {benchmark_2017.shape}")
    print(f"Shape of 2017 ETFs array: {etf_2017.shape}")

    # annualized tracking error for 2016
    assert benchmark_2016.shape[0] == etf_2016.shape[0]
    N = benchmark_2016.shape[0]

    annual_tracking_error_2016 = np.sqrt(252) * np.sqrt(
        np.sum((benchmark_2016 - etf_2016) ** 2, axis=0) / (N - 1)
    )
    # print(annual_tracking_error_2016[:])

    # annualized tracking error for 2017
    assert benchmark_2017.shape[0] == etf_2017.shape[0]
    N = benchmark_2017.shape[0]

    annual_tracking_error_2017 = np.sqrt(252) * np.sqrt(
        np.sum((benchmark_2017 - etf_2017) ** 2, axis=0) / (N - 1)
    )
    # print(annual_tracking_error_2017[:])

    # create dataframe of results from numpy arrays
    results = pd.DataFrame(
        data=[annual_tracking_error_2016, annual_tracking_error_2017],
        index=["annual_tracking_error_2016", "annual_tracking_error_2017"],
        columns=["Reliance", "Kotak", "HDFC", "UTI"],
    )
    print()
    print(results)


####################    QUESTION 2 functions  #################################


def read_data2():
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
    # print(f"\nPERIOD:")
    # print(f"start date: {date_start_str}")
    # print(f"end date: {date_end_str}")
    # print(f"shape before slicing: {df.shape}")
    df = df[date_start:date_end].copy()
    # print(f"shape after slicing: {df.shape}")

    # if first period normalize data
    if date_start == datetime(2016, 1, 1):
        # print("DATAFRAME SETTINGS")
        for asset in ("Nifty", "Junior", "Gold"):
            df[asset + "_norm_ret"] = df[asset] / df.iloc[0][asset]

        for asset, allocation in zip(("Nifty", "Junior", "Gold"), [0.5, 0.2, 0.3]):
            df[asset + "_allocation"] = df[asset + "_norm_ret"] * allocation

        for asset in ("Nifty", "Junior", "Gold"):
            df[asset + "_pos"] = df[asset + "_allocation"] * starting_fund

        df["Tot_pos"] = df["Nifty_pos"] + df["Junior_pos"] + df["Gold_pos"]

        df["daily_ret"] = df["Tot_pos"].pct_change(1)
        df_post = df
        return df_post

    else:
        # print("\nPORTFOLIO REBALANCING")
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

        df_post = pd.concat([df_previous, df_merged[1:]])
        df_post["daily_ret"] = df_post["Tot_pos"].pct_change(1)

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

            # UNCOMMENT THIS BLOCK OF CODE TO GRAPH THE PORTFOLIO RETURNS
            #############################################################
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
            # plt.show()
            #############################################################

            print(f"\nAnnualized Returns (CAGR) for {year_str}: {(cagr*100):.3f}%")
            # SHARPE ratio:
            print(f"Sharpe Ratio (SR) for {year_str}: {sharp_ratio:.3f}")

    return df_post


####################    QUESTION 3 functions  #################################


def read_data3(filepath, stock):
    df_2016 = pd.read_csv(
        f"{filepath}".format("16", stock),
        index_col=0,
        usecols=["Date", "Close Price"],
        parse_dates=True,
    )
    df_2016.rename(columns={"Close Price": stock}, inplace=True)
    df_2017 = pd.read_csv(
        f"{filepath}".format("17", stock),
        index_col=0,
        usecols=["Date", "Close Price"],
        parse_dates=True,
    )
    df_2017.rename(columns={"Close Price": stock}, inplace=True)

    return pd.concat([df_2016, df_2017], axis=0)


def indian_portfolio_ret(dataframe, indian_stocks):

    df = dataframe.copy()

    # normalized return for every stock
    for stock in indian_stocks:
        df[stock + "_norm_ret"] = df[stock] / df.iloc[0][stock]

    # allocation for every stock
    for stock, allocation in zip(
        indian_stocks, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ):
        df[stock + "_alloc"] = df[stock + "_norm_ret"] * allocation

    # position for every stock
    for stock in indian_stocks:
        df[stock + "_pos"] = df[stock + "_alloc"] * 100

    # keep only the necessary columns
    stocks = list(indian_stocks)
    col_to_keep = [stock + "_pos" for stock in stocks]
    df = df[col_to_keep]

    # total portfolio position
    df["Tot_pos"] = df.sum(axis=1)

    # daily return
    df["Y"] = df["Tot_pos"].pct_change(1)

    # UNCOMMENT THIS BLOCK OF CODE TO GRAPH THE PORTFOLIO RETURNS
    #############################################################
    # # plot portfolio and its components
    # fig, axs = plt.subplots(2)
    # # fig.tight_layout()
    # df[["Tot_pos"]].plot(
    #     ax=axs[0],
    #     color="black",
    #     grid=True,
    #     figsize=(20, 15),
    # )
    # df[col_to_keep].plot(
    #     ax=axs[1],
    #     grid=True,
    #     figsize=(20, 15),
    # )
    # plt.show()
    #############################################################

    cum_return = (df["Tot_pos"][-1] / df["Tot_pos"][0] - 1) * 100
    print()
    print("-" * 60)
    print(f"Cumulative Return for portfolio 2016-2017: {cum_return:.2f}%")
    print("-" * 60)
    print()

    return df["Y"]


def read_dependant_var(df):
    filepath1 = "./Q3-Data/Nifty ETF.xlsx"
    filepath2 = "./Q3-Data/Junior ETF.xlsx"

    df_nifty = pd.read_excel(
        filepath1,
        skiprows=3,
        names=["Date", "Nifty"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )

    df_junior = pd.read_excel(
        filepath2,
        skiprows=3,
        names=["Date", "Junior"],
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
    )
    df_nifty["nifty_daily_ret"] = df_nifty["Nifty"].pct_change(1)
    df_nifty.drop(columns=["Nifty"], inplace=True)
    df_junior["junior_daily_ret"] = df_junior["Junior"].pct_change(1)
    df_junior.drop(columns=["Junior"], inplace=True)

    return pd.concat([df, df_nifty, df_junior], axis=1)


def linear_regression(df):
    df_final = read_dependant_var(df)
    df_final.dropna(inplace=True)
    # independant variable is our chosen portfolio of daily returns
    Y = df_final["Y"]
    # dependant variables are Nifty and Junior ETFs
    R = df_final[["nifty_daily_ret", "junior_daily_ret"]]
    # adding a constant to the model
    R = sm.add_constant(R)

    model = sm.OLS(Y, R).fit()
    # predictions = model.predict(R)
    print_model = model.summary()
    print(print_model)


###########################################################################
if __name__ == "__main__":

    # Give the location of the file (OS: Ubuntu 20.04 LTS Unix distro)
    filepath1 = "./Q1-Data/NIFTY-TotalReturnsIndex.csv"
    filepath2 = "./Q1-Data/Reliance Nifty ETF.xlsx"
    filepath3 = "./Q1-Data/Kotak Nifty ETF.xlsx"
    filepath4 = "./Q1-Data/HDFC Nifty ETF.xlsx"
    filepath5 = "./Q1-Data/UTI Nifty ETF.xlsx"
    df = read_data()
    annualized_tracking_error(df)

    print("\n\nSolution for QUESTION 1")
    print("-" * 80)
    print(
        "Deliverable 1: Arrange the four funds (Reliance, Kotak, HDFC and UTI) in the increasing order of TE in 2016 and 2017"
    )
    print("Response for 2016: Reliance, Kotac, UTI, HDFC")
    print("Response for 2017: Reliance, Kotac, HDFC, UTI")

    print(
        "\nDeliverable 2: Out of the four funds, which ones have shown an increase in Annualized TE from 2016 to 2017?"
    )
    print("Response: None")

    print(
        "\nDeliverable 3: Out of the four funds, which ones have shown an decrease in Annualized TE from 2016 to 2017?"
    )
    print("Response: All of them")
    print("-" * 80)

    ############################################################################
    # Give the location of the file (OS: Ubuntu 20.04 LTS Unix distro)
    filepath1 = "./Q2-Data/Nifty ETF.xlsx"
    filepath2 = "./Q2-Data/Junior ETF.xlsx"
    filepath3 = "./Q2-Data/Gold ETF.xlsx"
    df = read_data2()
    eobq = pd.date_range(start="2016-01-01", end="2017-12-30", freq="BQ").to_list()

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

    print("\n\nSolution for QUESTION 2")
    print("-" * 80)
    print("Deliverable 1: Annualized returns for the strategy in 2016 and 2017")
    print("Annualized Returns for 2016 (CAGR): 7.139%")
    print("Annualized Returns for 2017 (CAGR): 23.875%")
    print()
    print("Deliverable 2: Sharpe Ratio for the strategy in 2016 and 2017")
    print("Sharpe Ratio for 2016: 0.748*")
    print("Sharpe Ratio for 2017: 3.364*")
    print(
        "* from the numerator of SR we should subtract the risk free rate. We assume it equals to zero since we do not have data"
    )
    print("-" * 80)
    print()

    ############################################################################
    filepath = "./Q3-Data/01-01-20{0}-TO-31-12-20{0}{1}EQN.csv"

    # list of chosen stocks from NSE (National Stock Exchange)
    """ 
    downloaded from:
    https://www1.nseindia.com/products/content/equities/equities/eq_security.htm
    """

    indian_stocks = (
        "TATASTEEL",
        "ONGC",
        "ITC",
        "TATAMOTORS",
        "HINDALCO",
        "COALINDIA",
        "RELIANCE",
        "NTPC",
        "POWERGRID",
        "SUNPHARMA",
    )

    df_final = pd.DataFrame()
    for stock in indian_stocks:
        df = read_data3(filepath, stock)
        df_final = pd.concat([df_final, df], axis=1)

    df_final.to_csv("./Q3-Data/final.csv")
    print(df_final.head())
    print(df_final.shape)
    df = indian_portfolio_ret(df_final, indian_stocks)
    linear_regression(df)

    print("\n\nSolution for QUESTION 3")
    print("-" * 80)
    print("Deliverable 1: Compute the coefficients β1 and β2:")
    print("β1 (Nifty coeff.) = .9633")
    print("β2 (Junior coeff.) = .2247")
    print("COMMENT: The returns of our portfolio is mainly explained by the Nifty ETF")
    print()
    print(
        "Deliverable 2: Compute of Portfolio return variation that is not explained by Nifty ETF and Junior ETF return variation"
    )
    print(
        "COMMENT: Since R^2 is 0.634 this implies that 36.6% of our portfolio return variation is unexplained by our model"
    )
    print("-" * 80)

    ############################################################################
