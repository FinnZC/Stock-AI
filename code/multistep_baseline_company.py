import numpy as np
import pandas as pd
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from company import Company


# Starts multi-step forecasting
class MultiStepBaselineCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string, n_lag,
                 n_seq):
        Company.__init__(self, name)
        self.scaler = None
        self.train_raw_series = self.get_share_prices(train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string,
                                                     start_delay=1)
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.train, self.test = self.preprocess_data()

    # convert time series into supervised learning problem
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # transform series into train and test sets for supervised learning
    def preprocess_data(self):
        if self.n_seq > len(self.train_raw_series):
            raise ValueError("There are no enough training data for", self.n_seq, "sequential forecast")
        # extract raw values
        train_raw_values = self.train_raw_series.values
        train_raw_values = train_raw_values.reshape(len(train_raw_values), 1)
        # transform into supervised learning problem X, y
        supervised_train = self.series_to_supervised(train_raw_values, self.n_lag, self.n_seq)
        supervised_train_values = supervised_train.values

        # extract raw values
        test_raw_values = self.test_raw_series.values
        test_raw_values = test_raw_values.reshape(len(test_raw_values), 1)
        # transform into supervised learning problem X, y
        supervised_test = self.series_to_supervised(test_raw_values, self.n_lag, self.n_seq)
        supervised_test_values = supervised_test.values
        """
        print("train size: ", len(self.train_raw_series), "  train supervised size: ", len(supervised_train_values))
        display(self.train_raw_series)
        display(supervised_train_values)
        print("test size: ", len(self.test_raw_series), "  test supervised size: ", len(supervised_test_values))
        display(self.test_raw_series)
        display(supervised_test_values)
        """
        return supervised_train_values, supervised_test_values

    # make a persistence forecast
    def persistence(self, last_ob):
        return [last_ob for i in range(self.n_seq)]

    # evaluate the persistence model
    def predict(self):
        predictions = pd.Series()
        # Index is datetime
        test_index = self.test_raw_series.index

        for i in range(len(self.test)):
            X, y = self.test[i, 0:self.n_lag], self.test[i, self.n_lag:]
            # print("X: ", X, "y: ", y)
            # make forecast
            pred = self.persistence(X[-1])
            # store the forecast
            predictions.at[test_index[i]] = pred
            # predictions.append(pred)
        return predictions

    # plot function for children classes, if run by parent, error would happen
    def plot(self, predictions):
        # line plot of observed vs predicted
        formatter = matplotlib.dates.DateFormatter('%d/%m/%Y')
        for test_date in predictions.index:
            # n_seq consecutive days
            x_axis = list()
            # The first day of test
            x_axis.append(test_date)
            for j in range(self.n_seq - 1):  # first one already added
                x_axis.append(self.date_by_adding_business_days(from_date=x_axis[-1], add_days=1))
            plt.plot(x_axis, predictions[test_date], ':', marker='*', color="blue", label="Predicted prices")

        plt.plot(self.train_raw_series.index, self.train_raw_series.values,
                 '-', marker=".", label="Actual prices (Training data)")

        # test data not always possible
        try:
            plt.plot(self.test_raw_series.index[:len(predictions)],
                     self.test_raw_series.values[:len(predictions)],
                     '-', marker=".", label="Actual prices (Test data)")
        except:
            # don't plot test data if not available
            print("Exception entered")
            pass
        # remove repeated legends
        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1
        plt.legend(handles, labels)
        ax = plt.gcf().axes[0]
        ax.xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate(rotation=25)
        plt.gcf().set_size_inches(15, 10)
        plt.xlabel("Time")
        plt.ylabel("Share Price ($)")
        plt.title("Stock price prediction for " + self.name)
        plt.show()

    # evaluate the RMSE for each forecast time step
    def score(self, predictions):
        # convert predictions to an appropriate list or arrays
        predictions = np.array([lst for lst in predictions.values])
        # display("test", self.test)
        # display("predicted", predictions)
        for i in range(self.n_seq):
            # first one is the test data and the next n_seq are predictions
            actual = self.test[:, 1:(self.n_lag + i + 1)]
            predicted = predictions[:, :(self.n_lag + i)]
            # display("actual", actual)
            # display("predicted", predicted)
            rmse = math.sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))