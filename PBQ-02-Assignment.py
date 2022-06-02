import sys
from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
print("\n","-"*80)
print(f"python version: {sys.version}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"seaborn version: {sns.__version__}")
print("\nWARNING: The assignment was written using Python on Ubuntu 20.04 LTS OS system (I am used to working with it). \nThere may be slight variations with regard to the syntax used with python scripts on Windows/MAC OS. \nPlease take it into consideration when evaluating the script.")
print("-"*80, "\n")


################################################################################
# Q1: Write a customized function to calculate the factorial of a number. Do not use any inbuilt functions. Create your own.

def factorial(n):
     return 1 if n in [0,1] else n*factorial(n-1)


################################################################################
# Q2: Write a Python program which iterates through a sequence of 50 randomly generated integers between 100 and 200, and derive two different series/arrays/lists from it, one which contains even multiples of 3 and the other which contains odd multiples of 5.

def generate_arrays():
    rand_integers= np.random.randint(100, 200, 50)
    print("random array of 50 numbers in [100, 200]:")
    print(rand_integers)
    print("\nArray of even numbers multiple of 3:")
    print(rand_integers[(rand_integers%3==0) & (rand_integers%2==0)])

    print("\nArray of odd numbers multiple of 5:")
    print(rand_integers[(rand_integers%5==0) & (rand_integers%2!=0)])


################################################################################
# Q3: Write a Python program to generate a sequence of 1000 numbers drawn from a standard uniform distribution and plot it on a chart. Compute its mean. Use the np.where clause to create a sequence of only numbers greater than 0.75 from the list. Plot a chart of those numbers. Check how many such numbers exist. Is it on expected lines? Comment. Please use np.random.seed(1234) at the beginning of your code. 

def gaussian_analysis():
    np.random.seed(1234)
    num_samples = 1000
    gaussian_dist=np.random.normal(loc=0, scale=1, size=num_samples)
    # print(gaussian_dist)

    plt.show()
    # Plot Gaussian Distribution
    sns.displot(data = gaussian_dist, kind="kde")
    plt.legend(labels= ["Normal Gaussian"])
    plt.show()

    # Mean of Gaussian Distribution
    print(f"mean of {num_samples} random generated gaussian samples is: {gaussian_dist.mean():.4f}")

    threshold =0.75
    greater_than_threshold= gaussian_dist[np.where(gaussian_dist>threshold)]
    # print(greater_than_threshold)

    # Plot Gaussian Distribution with threshold
    sns.displot(data = greater_than_threshold, kind="kde")
    plt.legend(labels= ["Gaussian with threshold"])
    plt.show()

    # Theoretical area under Gaussian for a specific threshold is Pr(x <= threshold). We need to calculate the opposite since we are sampling data greater than threshold: 
    #           Area = 1 -  Pr(x <= threshold)

    # Theoretical distribution using scipy "norm" formula
    area_greater_than_threshold = 1 - norm(loc=0, scale=1).cdf(threshold)

    print(f"\nTheoretical Gaussian area under x >= {threshold} is {(area_greater_than_threshold * 100):.2f}% of total area.")
    print(f"Count of generated Gaussian samples greater than {threshold} is: {len(greater_than_threshold)}")
    print(f"Area discrepancy between theoretical and generated distribution: {(abs(len(greater_than_threshold)/num_samples - area_greater_than_threshold)/(area_greater_than_threshold)) * 100:.2f}%")
    print("\nCOMMENT: with 1000 samples generated from the Gaussian distribution the theoretical discrepancy is slightly above 4%. It could be reduced by generating more samples.")


################################################################################
# Q4: Download the TCS.NS.csv file and read it into your Python environment.
    # (i) Which method/function of pandas displays the number of rows and columns? Use it to show the number of rows and columns for the data.
    # (ii) Use an appropriate method/function of pandas library to get the summary statistics of the data-frame.
    # (iii) What was the highest price reached by the TCS stock during this period?

def read_csv():
    print("\nUse of pd.read_csv() method to read the data.\n")
    data = pd.read_csv("TCS.NS.csv", index_col="Date", parse_dates=True, dayfirst=True)
    print(data.head())

    print("\nMAIN STATISTICS with describe() method")
    print(data.describe())

    print(f"\nHighest price reached during the period is: {data.max()[0]}")


################################################################################
# Q5: For the TCS.NS data downloaded for Q4,
    # (i) Fetch all the data from October 2017 using the .loc[] and save it in a new DataFrame called ‘df1’. 'From df1, display only the rows for which the value in the ‘Direction’ column is ‘UP’.
    # (ii) Add a new column to ‘df1’ containing the daily range (Close - Open) using a vectorized operation. 
    # (iii) Add another column to ‘df1’ containing the 3-day moving average of the Adj Close prices.
    # (iv) Use the appropriate method to delete all the rows containing NaN values from ‘df1’.

def tcs_data():
    # read the data from the excel file
    df = pd.read_excel("TCS.NS.xlsx", index_col="Date", parse_dates=True)

    # Fetch all the data from October 2017 using the .loc[] and save it in a new DataFrame called ‘df1’
    df1 = df.loc["Oct 2017":].copy()
    print("\ndata after October 2017:")
    print("------------------------")
    print(df1.head())

    # From df1, display only the rows for which the value in the ‘Direction’ column is ‘UP’.
    mask = df1["Direction"] == "UP"
    df1= df1[mask]
    print("\ndata with Direction = 'UP':")
    print("---------------------------")
    print(df1.head())

    # Add a new column to ‘df1’ containing the daily range (Close - Open) using a vectorized operation. 
    df1["daily range"] = df1["Close"] - df1["Open"]

    # Add another column to ‘df1’ containing the 3-day moving average of the Adj Close prices.
    df1["MA_3"] = df1["Adj Close"].rolling(window=3).mean()
    shape_before_del = df1.shape

    # Use the appropriate method to delete all the rows containing NaN values from ‘df1
    print("\nNumber of NaN by columns:")
    print("-------------------------")
    print(df1.isnull().sum())
    df1.dropna(axis=0, inplace=True)
    print("\nFINAL DATA:")
    print("-----------")
    print(df1.head(10))
    print(f"\nShape before deleting rows with NaN: {shape_before_del}")
    print(f"Shape after deleting rows with NaN: {df1.shape}")

 

if __name__=="__main__":

    # print("\nFUNCTION for Question 1:")
    # print("-"*80)
    # num=int(input("Enter the number: "))
    # print(f"The factorial of number {num} is: {factorial(num)}")
    
    # print("\n\nFUNCTION for Question 2:")
    # print("-"*80)
    # generate_arrays()

    # print("\n\nFUNCTION for Question 3:")
    # print("-"*80)
    # gaussian_analysis()

    print("\n\nFUNCTION for Question 4:")
    print("-"*80)
    read_csv()

    # print("\n\nFUNCTION for Question 5:")
    # print("-"*80)
    # tcs_data()
