import numpy as np
import pandas as pd

from datetime import date, timedelta

from sklearn.preprocessing import MinMaxScaler


class Bitcoin_LSTM:
    def __init__(self) -> None:
        super().__init__()
        self.scl = MinMaxScaler()
        self.X_quantity = None

    def get_prepared_data(self, timeseries, today, yesterday):
        timeseries[["Close-btc", "Close-sp500", "Close-dxy", "Close-gold"]] = self.scl.fit_transform(timeseries[[
            "Close-btc", "Close-sp500", "Close-dxy", "Close-gold"]])

        timeseries["Close-btc-output"] = timeseries["Close-btc"]
        timeseries["Close-btc-output"] = timeseries["Close-btc-output"].shift(-1)

        timeseries.loc[today, 'Close-btc-output'] = timeseries.loc[yesterday, 'Close-btc-output']

        array = timeseries.values

        return array

    def get_pred_train(self, array,  X_quantity, times):
        self.X_quantity = X_quantity

        mod = len(array) % self.X_quantity

        times = 7

        # deleting first-mod values to have /mod-zero array
        for i_ in range(mod):
            array = np.delete(array, 0, 0)

        # for splitting into train/test
        division = self.X_quantity * times
        split = len(array) - division

        predict = array[split:]
        train = array[:split]

        return predict, train

    def get_X_values(self, values):
        x = []
        ready_X = []
        COUNT = 1
        for i_ in values:
            x.extend(i_)
            if COUNT % self.X_quantity == 0:
                ready_X.append(x)
                x = []

            COUNT += 1

        ready_X = np.array(ready_X)
        return ready_X

    def get_Y_targets(self, targets):
        ready_Y = []
        for i_ in range(int(len(targets) / self.X_quantity)):
            i_ += 1
            ready_Y.append(targets[i_ * self.X_quantity - 1])

        ready_Y = np.array(ready_Y)
        return ready_Y

    def get_array(self, g):
        g = np.insert(g, [1], .4, axis=1)
        g = np.insert(g, [2], .4, axis=1)
        g = np.insert(g, [3], .4, axis=1)

        array_ = self.scl.inverse_transform(g)
        array_ready = []
        for i in range(len(array_[:, :1])):
            array_ready.append(array_[i, :1][0])

        return array_ready

    def get_forecast(self, timeseries, pred, btc):
        t = timeseries.reset_index()
        timestamp = pd.DataFrame()
        timestamp['data'] = t['index'].copy()

        DAY = 0
        ds = []
        y_actual = []
        tomorrow = date.today() + timedelta(days=1)
        tomorrow = tomorrow.strftime('%Y-%m-%d')

        for i_ in range(len(pred)):
            if i_ == (len(pred) - 1):
                ds.append(tomorrow)
                y_actual.append(0)
            else:
                DAY = DAY + 3
                ds.append(timestamp.iloc[DAY][0].strftime('%Y-%m-%d'))
                y_actual.append(btc.iloc[DAY])

        d = {"ds": ds, "yhat": pred, 'y_actual': y_actual}
        forecast = pd.DataFrame(d)
        return forecast