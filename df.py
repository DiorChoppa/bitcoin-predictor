import pandas as pd
import yfinance as yf
from datetime import datetime

dd = yf.download('BTC-USD', start='2015-08-07', end=datetime.today().strftime('%Y-%m-%d'))
path='data/BTC-USD.csv'
dd.to_csv(path)

df = pd.read_csv('data/BTC-USD.csv')

# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# df['Date'] = df['Date'][:10]

# print(df.iloc[:, :0])

for i in range(0, df['Date'].count()):
    data = df['Date'][i].split(" ")
    df['Date'][i] = data[0]
    # df['Date'][i].replace(data[0])

# df['Date'] = df['Date'].astype(str)

# print(df['Date'].index[-1])
# for i in df:
#     print(i)
# last_day = df['Date'].index[-1].strftime('%Y-%m-%d')
# if last_day == datetime.today().strftime('%Y-%m-%d'):
#     pass
# else:
del df[df.columns[0]]
print('OOOOOOOOOOO\n')
for i in df:
    print(i)
df.to_csv(path)

# print(df.columns)