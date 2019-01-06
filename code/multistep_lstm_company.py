import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from company import Company


class MultiStepLSTMCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons):
        Company.__init__(self, name)
        self.scaler = None
        self.train_raw_series = self.get_share_prices(train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string,
                                                     start_delay=1)
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.train, self.test = self.preprocess_data()

    # create a differenced series
    def difference(self, series, source, interval=1):
        diff = list()
        # First item is special case because we use the difference of the last training pair to predict the first test price
        if source == "test":
            diff.append(self.train_raw_series.values[-1] - self.train_raw_series.values[-2])
        for i in range(1, len(series)):
            value = series[i] - series[i - 1]
            diff.append(value)
        # Last item is special case because there is no next value thus the diff is
        # 1 size shorter than the original test_raw. We fix this by adding an additional item
        if source == "test":
            diff.append(0)  # placeholder for the last prediction, not used in anyway
        return pd.Series(diff)

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

    # scale train and test data to [-1, 1]
    def scale(self, train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # transform series into train and test sets for supervised learning
    def preprocess_data(self):
        if self.n_seq > len(self.train_raw_series):
            raise ValueError("There are no enough training data for", self.n_seq, "sequential forecast")
        # extract raw values
        train_raw_values = self.train_raw_series.values
        train_raw_values = train_raw_values.reshape(len(train_raw_values), 1)
        # transform into supervised learning problem X, y
        supervised_train = self.series_to_supervised(train_raw_values, self.n_lag, self.n_seq)
        train = supervised_train.values

        # extract raw values
        test_raw_values = self.test_raw_series.values
        test_raw_values = test_raw_values.reshape(len(test_raw_values), 1)
        # transform into supervised learning problem X, y
        supervised_test = self.series_to_supervised(test_raw_values, self.n_lag, self.n_seq)
        test = supervised_test.values

        # transform the scale of the data
        scaler, train_scaled, test_scaled = self.scale(train, test)
        self.scaler = scaler
        """
        print("train size: ", len(self.train_raw_series), "  train supervised size: ", len(supervised_train_values))
        display(self.train_raw_series)
        display(supervised_train_values)
        print("test size: ", len(self.test_raw_series), "  test supervised size: ", len(supervised_test_values))
        display(self.test_raw_series)
        display(supervised_test_values)
        """
        return train_scaled, test_scaled

        # fit the model

    def train(self):
        print("Fitting the model")
        self.lstm_model = self.fit_lstm(self.train_scaled)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = self.train_scaled[:, 0].reshape(len(self.train_scaled), 1, 1)
        self.lstm_model.predict(train_reshaped, batch_size=n_batch)
        print("Finished fitting the model")

    # fit an LSTM network to training data
    def fit_lstm(self, train):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:self.n_lag], train[:, self.n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # design network
        model = Sequential()
        model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        for i in range(self.n_epoch):
            model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make one forecast with an LSTM,
    def forecast_lstm(self, model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # evaluate the persistence model
    def make_forecasts(self, model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted

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
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))

