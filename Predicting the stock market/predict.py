### Predicting the stock market ###

"""
In this project, I'll be working with data from the S&P500 Index. 
The S&P500 is a stock market index. 

Some companies are publicly traded, which means that anyone can buy and sell their shares on the open market. 
A share entitles the owner to some control over the direction of the company, and to some percentage (or share) of the earnings of the company. 
When you buy or sell shares, it's common to say that you're trading a stock.

Indexes aggregate the prices of multiple stocks together, and allowing to see how the market as a whole is performing.

I'll be using historical data on the price of the S&P500 Index to make predictions about future prices.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

### Read the Data

market = pd.read_csv("sphist.csv")
market["Date"]=pd.to_datetime(market["Date"])
market=market.sort_values(by="Date", ascending=True)
market=market.reset_index(drop=True)

### Generating Indicators

list_365_price=[]
list_365_std_price=[]
list_ratio=[]
list_365_volume=[]
list_ratio_volume=[]
for index, row in market.iterrows():
    if index>365:
        avg_365_price=market["Close"].iloc[index-365:index].mean()
        stdev_365_price=market["Close"].iloc[index-365:index].std()
        ratio_365_5=market["Close"].iloc[index-5:index].mean()/avg_365_price
        avg_volume_365=market["Volume"].iloc[index-365:index].mean()
        ratio_365_5_volume=market["Volume"].iloc[index-5:index].mean()/avg_volume_365
    else:
        avg_365_price=0
        stdev_365_price=0
        ratio_365_5=0
        avg_volume_365=0
        ratio_365_5_volume=0
    list_365_price.append(avg_365_price)
    list_365_std_price.append(stdev_365_price)
    list_ratio.append(ratio_365_5)
    list_365_volume.append(avg_volume_365)
    list_ratio_volume.append(ratio_365_5_volume)
ser_365_price=pd.Series(list_365_price, name="365_mean_price")
ser_365_std_price=pd.Series(list_365_std_price, name="365_std_price")
ser_ratio=pd.Series(list_ratio, name="5_365_ratio")
ser_365_volume=pd.Series(list_365_volume, name="365_mean_volume")
ser_ratio_volume=pd.Series(list_ratio_volume, name="5_365_ratio_volume")

market=pd.concat([market,ser_365_price,ser_365_std_price,ser_ratio,ser_365_volume,ser_ratio_volume], axis=1)

market=market[market["Date"] > datetime(year=1951, month=1, day=2)]
market=market[market["365_mean_price"]!=0]

market=market.reset_index(drop=True)

### Train and Test the Model

train=market[market["Date"]<datetime(year=2013, month=1, day=1)]
test=market[market["Date"]>=datetime(year=2013, month=1, day=1)]

train_cols=["365_mean_price", "365_std_price", "5_365_ratio", "365_mean_volume", "5_365_ratio_volume"]
target="Close"
lr = LinearRegression()

lr.fit(train[train_cols], train[target])
predictions_test = lr.predict(test[train_cols])
test_mae=mean_absolute_error(test[target], predictions_test)
print(test_mae)

### Making Predictions Just One Day Ahead

lr2 = LinearRegression()
start_index=test.iloc[0].name
end_index=test.iloc[-1].name
it_test_mae_list=[]
for i in range(start_index, end_index):
    it_train=market.iloc[:i-1]
    it_test=pd.DataFrame(market.iloc[i]).transpose()
    lr2.fit(it_train[train_cols], it_train[target])
    predictions_it_test = lr2.predict(it_test[train_cols])
    it_test_mae=mean_absolute_error(it_test[target], predictions_it_test)
    print(it_test_mae)
    it_test_mae_list.append(it_test_mae)
mean_it_test_mae=np.mean(it_test_mae_list)

print("Mean of all the found maes: ", mean_it_test_mae)

