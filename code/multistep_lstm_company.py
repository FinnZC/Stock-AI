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
from time import time


class MultiStepLSTMCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons):
        Company.__init__(self, name)
        self.scaler = None
        self.lstm_model = None
        self.train_raw_series = self.get_filtered_series(self.share_prices_series,
                                                         train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_filtered_series(self.share_prices_series, train_end_test_start_date_string,
                                                        test_end_date_string, start_delay=1)

        self.train_start_date_string = train_start_date_string
        self.train_end_test_start_date_string = train_end_test_start_date_string
        self.test_end_date_string = test_end_date_string
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.train_scaled, self.test_scaled = None, None
        self.preprocess_data()
        self.time_taken_to_train = 0

    def update_train_test_set(self, start_train, end_train_start_test, end_test):
        self.train_start_date_string = start_train
        self.train_end_test_start_date_string = end_train_start_test
        self.test_end_date_string = end_test
        self.preprocess_data()

    def preprocess_data(self):
        data_series = self.train_raw_series.append(self.test_raw_series)
        display("data series", data_series)
        diff_series = self.difference(data_series)
        # display("difference series", diff_series)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)

        supervised_pd = self.timeseries_to_supervised(diff_values, self.n_lag, self.n_seq)
        # display("supervised", supervised_pd)

        cutoff = len(self.train_raw_series) - self.n_seq - 1
        train_supervised_values = supervised_pd.values[:cutoff - self.n_lag + 1]
        # display("train supervised", train_supervised_values)
        test_supervised_values = supervised_pd.values[cutoff + self.n_seq - self.n_lag:]
        # display("test supervised", test_supervised_values)
        self.scaler, scaled_train_supervised, scaled_test_supervised = self.scale(train_supervised_values,
                                                                                  test_supervised_values)
        display("scaled train supervised", scaled_train_supervised)
        display("scaled test supervised", scaled_test_supervised)

        self.train_scaled, self.test_scaled = scaled_train_supervised, scaled_test_supervised

    # create a differenced series
    def difference(self, series):
        diff_series = pd.Series()
        index = series.index
        for i in range(1, len(index)):
            diff_series.at[index[i]] = series[index[i]] - series[index[i - 1]]
        return diff_series

    # convert time series into supervised learning problem
    def timeseries_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        print("data type", type(data))
        n_vars = 1 if (type(data) is list or isinstance(data, pd.Series)) else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        column_names = list(df.columns.values)
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (column_names[j], i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (column_names[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (column_names[j], i)) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # scale train and test data to [-1, 1]
    def scale(self, train_raw, test_raw):
        # fit scaler with 1 Dimensional array data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_train_data = train_raw.reshape(train_raw.size, 1)
        # display("fit scaler with train data", scaler_train_data)
        scaler = scaler.fit(scaler_train_data)
        # transform train
        train_scaled = scaler.transform(scaler_train_data)
        train_scaled = train_scaled.reshape(train_raw.shape[0], train_raw.shape[1])
        # display("train_scaled", train_scaled)
        # transform test
        test_data = test_raw.reshape(test_raw.size, 1)
        test_scaled = scaler.transform(test_data)
        test_scaled = test_scaled.reshape(test_raw.shape[0], test_raw.shape[1])
        # display("test_scaled", test_scaled)

        return scaler, train_scaled, test_scaled

    # fit the model
    def train(self):
        print("Fitting the model")
        start_time = time()
        self.lstm_model = self.fit_lstm(self.train_scaled)
        self.reset()
        self.time_taken_to_train = time() - start_time
        print("Finished fitting the model, time taken to train: %.1f s" % self.time_taken_to_train)

    def reset(self):
        # forecast the entire training dataset to build up state for forecasting
        # reshape training into [samples, timesteps, features]
        self.lstm_model.reset_states()
        X, y = self.train_scaled[:, 0:self.n_lag], self.train_scaled[:, self.n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.lstm_model.predict(X, batch_size=self.n_batch)

    # fit an LSTM network to training data
    def fit_lstm(self, train):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:self.n_lag], train[:, self.n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        display("train X data", X)
        display("train y data", y)
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

    # make a one-step forecast standalone
    def forecast_lstm_one_step(self):
        self.reset()
        predictions = pd.Series()
        train_index = self.train_raw_series.index

        X = np.array(self.train_scaled[len(self.train_scaled) - 1, self.n_seq:])
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
        self.reset()
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

        # display("predictions before inverse transform", predictions)
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
        test_values = self.test_raw_series.values
        actual = list()
        for i in range(len(test_values) - self.n_seq + 1):
            next_days_values = test_values[i: i + self.n_seq]
            actual.append(next_days_values)
        actual = np.array(actual)
        # display("actual", actual)

        predictions = np.array(predictions.tolist())
        # display("predicted", predictions)

        if metric == "rmse":
            rmses = list()
            for i in range(self.n_seq):
                # first one is the test data and the next n_seq are predictions
                rmse = math.sqrt(mean_squared_error(actual[:, i], predictions[:, i]))
                print('t+%d RMSE: %f' % ((i + 1), rmse))
                rmses.append(rmse)

            return rmses

        elif metric == "trend":
            # first case is special case since the last data input from the training data is used
            price_1_day_before = self.train_raw_series[-1]
            index = self.test_raw_series.index
            trends = list()
            for i in range(self.n_seq):
                print("\nCalculating trend score for ", i + 1)
                correct_counts = 0
                for j in range(len(predictions)):
                    actual = self.test_raw_series[index[j + i]]
                    if actual > price_1_day_before:
                        true_trend = "up"
                    elif actual < price_1_day_before:
                        true_trend = "down"
                    else:
                        true_trend = "neutral"

                    if predictions[j, i] > price_1_day_before:
                        predicted_trend = "up"
                    elif predictions[j, i] < price_1_day_before:
                        predicted_trend = "down"
                    else:
                        predicted_trend = "neutral"

                    if true_trend == predicted_trend:
                        correct_counts += 1
                    print("Price 1 day before: ", price_1_day_before)
                    print("Actual price: ", actual, " | Predicted price: ", predictions[j, i])
                    print("Actual trend: ", true_trend, " | Predicted trend: ", predicted_trend)
                    # next day
                    price_1_day_before = actual
                price_1_day_before = self.test_raw_series[index[i]]
                print("Correct counts: ", correct_counts, "  Size of test set:", len(self.test_raw_series))
                trends.append(correct_counts / len(self.test_raw_series))
            return trends
        else:
            print(metric, " is not an valid metric. Return NONE")
            return None

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = list()
        new = forecast[0] + last_ob
        inverted.append(new)
        print("Inverse difference Pred: ", forecast[0], "  + Reference Price:", last_ob, " = ", new)
        last_ob = new
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            new = forecast[i] + inverted[i - 1]
            inverted.append(new)
            print("Inverse difference Pred: ", forecast[i], "  + Reference Price:", last_ob, " = ", new)
            last_ob = new

        print("Final inverted values: ", inverted)
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, series, predictions, n_test):
        # walk-forward validation on the test data
        inverted_predictions = pd.Series()
        pred_index = predictions.index
        for i in range(len(predictions)):
            # create array from forecast
            pred = array(predictions[i])
            pred = pred.reshape(len(pred), 1)
            # invert scaling
            inv_scale = self.scaler.inverse_transform(pred)
            # inv_scale = inv_scale[0, :]
            print("Inverse scale  Original Pred: ", pred, "   After Scaling: ", inv_scale)
            # invert differencing
            # -1 to get the t-1 price
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted_predictions.at[pred_index[i]] = inv_diff
        return inverted_predictions