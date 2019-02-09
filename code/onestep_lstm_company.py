import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from company import Company


class OneStepLSTMCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_epochs, n_batch, n_neurons):
        Company.__init__(self, name)
        self.lstm_model = None
        self.scaler = None
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.train_raw_series = self.get_share_prices(train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string,
                                                     start_delay=1)
        self.train_scaled, self.test_scaled = self.preprocess_data()
        # same as test_raw_series with the addition of the last element of train_raw_series
        # Add the last training sample to the start of the test raw series so the invert differnce can work
        self.invert_difference_series_values = [self.train_raw_series.values[-1]] + self.test_raw_series.values.tolist()

    # create a differenced series
    def difference(self, series, source):
        diff = list()
        # First item is special case because we use the difference of the last training pair to predict the first test price
        if source == "test":
            diff.append(self.train_raw_series.values[-1] - self.train_raw_series.values[-2])
            diff.append(self.test_raw_series.values[0] - self.train_raw_series.values[-1])

        for i in range(1, len(series)):
            value = series[i] - series[i - 1]
            diff.append(value)

        return pd.Series(diff)

    # adapted from https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
    def timeseries_to_supervised(self, data, lag=1):
        df = pd.DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

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

    def preprocess_data(self):
        # transform data to be stationary
        train_diff_values = self.difference(self.train_raw_series.values, "train")
        test_diff_values = self.difference(self.test_raw_series.values, "test")

        # transform data to be supervised learning
        train_supervised_pd = self.timeseries_to_supervised(train_diff_values, 1).iloc[1:]
        train = train_supervised_pd.values

        # removes first row because it is not relevant
        test_supervised_pd = self.timeseries_to_supervised(test_diff_values, 1).iloc[1:]
        test = test_supervised_pd.values

        # transform the scale of the data
        scaler, train_scaled, test_scaled = self.scale(train, test)
        self.scaler = scaler

        """


        print("size of diff train data: ", len(train_diff_values))
        display(train_diff_values)
        print("size of supervised train data: ", len(train))
        display(train)



        print("size of diff test data: ", len(test_diff_values))
        display(test_diff_values)
        print("size of supervised test data: ", len(test))
        display(test)
        print("size of train_raw data: ", len(self.train_raw_series))
        display(self.train_raw_series)
        print("size of test_raw data: ", len(self.test_raw_series))
        display(self.test_raw_series)

        print("size of supervised train_scaled data: ", len(train_scaled))
        print(train_scaled)
        print("size of supervised test_scaled data: ", len(test_scaled))
        print(test_scaled)
        """
        display("raw train", self.train_raw_series)
        display("raw test", self.test_raw_series)
        # display("diff train", train_diff_values)
        display("scaled train", train_scaled)
        # display("supervised train", train_supervised_pd)

        # display("diff test", test_diff_values)
        display("scaled test", test_scaled)
        # display("supervised test", test_supervised_pd)

        return train_scaled, test_scaled

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
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(self.n_epochs):
            model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
        model.reset_states()
        return model

    # make a one-step forecast within test
    def forecast_lstm(self, X):
        X = X.reshape(1, 1, len(X))
        pred = self.lstm_model.predict(X, batch_size=self.n_batch)
        return pred[0, 0]

    # make a one-step forecast standalone
    def forecast_lstm_one_step(self):
        predictions = pd.Series()
        yesterday_price_change = self.train_scaled[-1, -1:]
        print("Yesterday price: ", yesterday_price_change)

        X = yesterday_price_change
        print("X: ", X)
        # predicting the change
        pred_price = self.forecast_lstm(X)
        # invert scaling
        pred_price = self.invert_scale(X, pred_price)
        # invert differencing
        pred_price = pred_price + self.train_raw_series.values[-1]

        # Prediction for the next business working day
        predictions.at[self.date_by_adding_business_days(
            from_date=self.train_raw_series.index[-1], add_days=1)] = pred_price
        print(predictions)
        return predictions

    def predict(self):
        # walk-forward validation on the test data
        predictions = pd.Series()
        # Index is datetime
        test_index = self.test_raw_series.index
        # predict the fist share price after the last share price in the training data
        # pred = self.forecast_lstm(1, self.train_scaled[i, 0:-1])
        for i in range(len(self.test_scaled)):
            # make one-step forecast
            X, y = self.test_scaled[i, 0:-1], self.test_scaled[i, -1]
            print("X: ", X, "y: ", y)
            pred = self.forecast_lstm(X)
            print("Pred: ", pred)
            # invert scaling
            pred = self.invert_scale(X, pred)
            # invert differencing
            pred = self.inverse_difference(pred, i)
            # store forecast
            predictions.at[test_index[i]] = pred

        # expected = self.invert_scale(X, y)
        # expected = self.inverse_difference(self.test_raw_series, expected, len(self.test_scaled)-i)
        # exp = self.test_raw_series[test_index[i]]
        # print('Predicted=%f, Expected Raw = %f' % (pred, exp))

        print("predictions", predictions)
        return predictions

    # invert differenced value
    def inverse_difference(self, pred, index=1):
        # print("interval", interval)
        print("Inverse difference  Pred: ", pred, " + Reference Price: ", self.invert_difference_series_values[index])
        return pred + self.invert_difference_series_values[index]

    # inverse scaling for a forecasted value
    def invert_scale(self, X, pred):
        new_row = [x for x in X] + [pred]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        print("Inverse scale  Original Pred: ", pred, "   After Scaling: ", inverted[0, -1])
        return inverted[0, -1]
