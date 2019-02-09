import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from numpy import array
from company import Company


class MultiStepLSTMCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons):
        Company.__init__(self, name)
        self.scaler = None
        self.lstm_model = None
        self.train_raw_series = self.get_share_prices(train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string,
                                                     start_delay=1)
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.train_scaled, self.test_scaled = self.preprocess_data()

    def preprocess_data(self):

        # transform data to be stationary
        train_diff_series = self.difference(self.train_raw_series.values, "train")
        train_diff_values = train_diff_series.values
        train_diff_values = train_diff_values.reshape(len(train_diff_values), 1)

        test_diff_series = self.difference(self.test_raw_series.values, "test")
        test_diff_values = test_diff_series.values
        test_diff_values = test_diff_values.reshape(len(test_diff_values), 1)

        # transform the scale of the data
        scaler, train_scaled, test_scaled = self.scale(train_diff_values, test_diff_values)
        train_scaled = train_scaled.reshape(len(train_scaled), 1)
        test_scaled = test_scaled.reshape(len(test_scaled), 1)
        self.scaler = scaler

        # transform data to be supervised learning
        train_supervised_pd = self.timeseries_to_supervised(train_scaled, self.n_lag, self.n_seq)

        train = train_supervised_pd.values

        # removes first row because it is not relevant
        test_supervised_pd = self.timeseries_to_supervised(test_scaled, self.n_lag, self.n_seq)
        test = test_supervised_pd.values
        display("raw train", self.train_raw_series)
        display("raw test", self.test_raw_series)
        # display("diff train", train_diff_values)
        # display("scaled train", train_scaled)
        display("supervised train", train_supervised_pd)

        # display("diff test", test_diff_values)
        # display("scaled test", test_scaled)
        display("supervised test", test_supervised_pd)
        return train, test

    # create a differenced series
    def difference(self, series, source="train"):
        diff = list()
        # First item is special case because we use the difference of the last training pair to predict the first test price
        if source == "test":
            diff.append(self.train_raw_series.values[-1] - self.train_raw_series.values[-2])
            diff.append(self.test_raw_series.values[0] - self.train_raw_series.values[-1])

        for i in range(1, len(series)):
            value = series[i] - series[i - 1]
            diff.append(value)

        return pd.Series(diff)

    # convert time series into supervised learning problem
    def timeseries_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
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

    # scale train and test data to [-1, 1]
    def scale(self, train, test):
        # fit scaler
        display(train)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # fit the model
    def train(self):
        print("Fitting the model")

        self.lstm_model = self.fit_lstm(self.train_scaled)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = self.train_scaled[:, 0].reshape(len(self.train_scaled), 1, 1)
        self.lstm_model.predict(train_reshaped, batch_size=self.n_batch)
        print("Finished fitting the model")

    # fit an LSTM network to training data
    def fit_lstm(self, train):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:self.n_lag], train[:, self.n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        display(X)
        display(y)
        # design network
        model = Sequential()
        model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        for i in range(self.n_epochs):
            model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make one forecast with an LSTM,
    def forecast_lstm(self, X):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = self.lstm_model.predict(X, batch_size=self.n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # plot function for children classes, if run by parent, error would happen
    def plot(self, predictions):
        # line plot of observed vs predicted
        formatter = matplotlib.dates.DateFormatter('%d/%m/%Y')
        # display(predictions.index)
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
            plt.plot(self.test_raw_series.index,
                     self.test_raw_series.values,
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

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        display("last_ob:", last_ob, "   forecast:", forecast)

        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, series, predictions, n_test):
        # walk-forward validation on the test data
        inverted_predictions = pd.Series()
        pred_index = predictions.index
        for i in range(len(predictions)):
            # create array from forecast
            pred = array(predictions[i])
            pred = pred.reshape(1, len(pred))
            # invert scaling
            inv_scale = self.scaler.inverse_transform(pred)
            inv_scale = inv_scale[0, :]
            # invert differencing
            # -1 to get the t-1 price
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted_predictions.at[pred_index[i]] = inv_diff
        return inverted_predictions

    # make a one-step forecast standalone
    def forecast_lstm_one_step(self):
        predictions = pd.Series()
        train_index = self.train_raw_series.index

        X = np.array([self.train_scaled[len(self.train_scaled) - 1, -1]])
        print("X: ", X, "y: ?")

        pred = self.forecast_lstm(X)
        # store forecast
        predictions.at[self.date_by_adding_business_days(train_index[-1], 1)] = pred

        # print("Predictions before inverse transform")
        # inverse transform
        predictions = self.inverse_transform(self.train_raw_series, predictions, len(self.train_raw_series))
        return predictions

    # evaluate the persistence model
    def predict(self):
        # walk-forward validation on the test data
        predictions = pd.Series()
        # Index is datetime
        test_index = self.test_raw_series.index
        for i in range(len(self.test_scaled)):
            # make multi-step forecast
            X, y = self.test_scaled[i, 0:self.n_lag], self.test_scaled[i, self.n_lag:]
            print("X: ", X, "y: ", y)
            pred = self.forecast_lstm(X)
            print("Prediction: ", pred)
            # store forecast
            predictions.at[test_index[i]] = pred

        display("predictions before inverse transform", predictions)
        # inverse transform
        predictions = self.inverse_transform(self.train_raw_series.append(self.test_raw_series), predictions,
                                             len(self.test_raw_series))
        print("Predictions after inverse transform")
        display(predictions)
        return predictions

    # evaluate the RMSE for each forecast time step
    def score(self, metric, predictions):
        # convert actual tests and predictions to an appropriate list or arrays
        # construct list of rows
        # first item is test data for the next days, hence not taken into account to measure the prediction
        test_values = multi_step_lstm.test_raw_series.values
        actual = list()
        for i in range(len(test_values) - self.n_seq + 1):
            next_days_values = test_values[i + self.n_lag - 1: i + self.n_seq]
            actual.append(next_days_values)
        actual = np.array(actual)
        display("actual", actual)

        predictions = np.array(predictions.tolist())
        display("predicted", predictions)

        if metric == "rmse":
            rmses = list()
            for i in range(self.n_seq):
                # first one is the test data and the next n_seq are predictions
                rmse = math.sqrt(mean_squared_error(actual[:, i], predictions[:, i]))
                print('t+%d RMSE: %f' % ((i + 1), rmse))
                rmses.append(rmse)

            return rmses
