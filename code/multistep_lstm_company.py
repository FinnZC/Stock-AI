import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D


from sklearn.metrics import mean_squared_error
from numpy import array
from company import Company
from time import sleep
from time import time
import pickle


class MultiStepLSTMCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_neurons, n_batch=None, tech_indicators=[], model_type="vanilla"):
        Company.__init__(self, name)
        self.scaler = None
        self.lstm_model = None
        self.train_scaled, self.test_scaled = None, None
        self.supervised_pd = None
        self.raw_pd = None
        self.train_raw_series, self.test_raw_series = None, None
        self.model_type = model_type
        if tech_indicators == "all":
            self.input_tech_indicators_list = self.all_tech_indicators
        else:
            self.input_tech_indicators_list = tech_indicators
        self.train_start_date_string = train_start_date_string
        self.train_end_test_start_date_string = train_end_test_start_date_string
        self.test_end_date_string = test_end_date_string
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.time_taken_to_train = 0
        self.preprocess_data()

    def add_tech_indicators_dataframe(self, price_series, indicators):
        combined = price_series
        for ind in indicators:
            while True:  # try again until success
                try:
                    ind_series = self.get_indicator(ind)
                    combined = pd.concat([combined, ind_series], axis=1)
                    break
                except KeyError:
                    # Could be that API has reached its limit
                    print("Retrying to download indicator", ind, "due to API limit 5 calls per minute and 500 calls per day")
                    sleep(20)
                    pass
        return combined

    def get_indicator(self, ind_name):
        if self.raw_pd is not None:
            if ind_name.upper() not in self.raw_pd.columns:
                print("Downloading ", ind_name)
                data, meta_data = getattr(self.tech_indicators, "get_" + ind_name)(self.name, interval="daily")
                #print(meta_data)
                data.index = pd.to_datetime(data.index)
                data.rename(columns={data.columns[0]: ind_name.upper()}, inplace=True)
            else:
                #print("Add from existing raw_pd ", ind_name)
                data = self.raw_pd[ind_name.upper()]
        else:
            print("Downloading ", ind_name)
            data, meta_data = getattr(self.tech_indicators, "get_" + ind_name)(self.name, interval="daily")
            data.index = pd.to_datetime(data.index)
            data.rename(columns={data.columns[0]: ind_name.upper()}, inplace=True)

        return data

    def update_raw_pd(self):
        print("Updating all series: share prices", ", ".join(self.input_tech_indicators_list))
        self.raw_pd = None
        self.preprocess_data()

    def preprocess_data(self):
        print("Preprocessing the data")
        start_time = time()
        try:
            data = pd.read_csv("raw_data/" + self.name + "_raw_pd.csv", index_col=0)
            data.index = pd.to_datetime(data.index)
            self.raw_pd = data
            self.share_prices_series = data["Share Price"]
            print("Retrieved price series and raw pd from disk")
        except FileNotFoundError:
            print("No existing data exist for this company so start downloading")
            while True:
                try:
                    self.share_price_series, metadata = self.time_series.get_daily_adjusted(symbol=self.name, outputsize='full')
                    break
                except KeyError:
                    # Could be that API has reached its limit
                    print("Retrying to download price series")
                    sleep(20)
                    pass
                # Convert index of the DataFrame which is in the date string format into datetime
            self.share_price_series.index = pd.to_datetime(self.share_price_series.index)

        if "price" in self.input_tech_indicators_list:
            price_in_ind_list = True
            self.input_tech_indicators_list.remove("price")
        else:
            price_in_ind_list = False

        #display("price data series", len(price_series), price_series)
        if len(self.input_tech_indicators_list) > 0:
            # add additional technical indicators
            combined = self.add_tech_indicators_dataframe(self.share_prices_series, self.input_tech_indicators_list)
        else:
            combined = self.share_prices_series


        self.raw_pd = combined

        #display("train raw series", self.train_raw_series)
        #display("test raw series", self.test_raw_series)

        supervised_pd = self.timeseries_to_supervised(combined, self.n_lag, self.n_seq)

        # display("supervised", supervised_pd)
        # delete unnecessary variables for prediction except price (should be var1)
        supervised_pd = self.drop_irrelevant_y_var(supervised_pd, price_in_ind_list)
        #display("supervised_pd original", supervised_pd)
        supervised_pd = self.difference(supervised_pd)
        #display("supervised_pd after differencing", supervised_pd)

        supervised_pd = self.get_filtered_series(supervised_pd, self.train_start_date_string, self.test_end_date_string)


        self.supervised_pd = supervised_pd
        #display("supervised filtered pd ", supervised_pd)
        train_raw_index = self.get_filtered_series(supervised_pd, self.train_start_date_string,
                                                         self.train_end_test_start_date_string).index
        self.train_raw_series = self.share_prices_series[train_raw_index]

        test_raw_index = self.get_filtered_series(supervised_pd, self.train_end_test_start_date_string,
                                                         self.test_end_date_string, start_delay=1).index
        self.test_raw_series = self.share_prices_series[test_raw_index]

        cutoff = len(self.train_raw_series)


        # display("train supervised values", train_supervised_values)
        train_supervised_values = supervised_pd.values[:cutoff]
        test_supervised_values = supervised_pd.values[cutoff:]


        #display("test supervised values", test_supervised_values)

        #display("filtered train values", supervised_pd)

        self.scaler, scaled_train_supervised, scaled_test_supervised = self.scale(train_supervised_values,
                                                                                  test_supervised_values)
        #display("scaled train supervised", scaled_train_supervised.shape)
        #display("scaled test supervised", scaled_test_supervised)
        self.train_scaled, self.test_scaled = scaled_train_supervised, scaled_test_supervised
        print("Preprocessed data in ", (time() - start_time)/60, "mins")

        if self.n_batch == "full_batch":
            self.n_batch = len(self.train_raw_series)
        elif self.n_batch == "online":
            self.n_batch = 1
        elif self.n_batch == "half_batch":
            half = int(len(self.train_raw_series)/2)
            for i in range(half)[::-1]:
                if len(self.train_raw_series) % i == 0:
                    self.n_batch = i
                    break
        else:
            raise ValueError("n_batch is not full_batch, half_batch, nor online. Must be one of them!")

    def update_train_test_set(self, start_train, end_train_start_test, end_test):
        print("Update the training and testing set with the specified dates: "
              "train data from %s to %s and test data from %s to %s"
              % (start_train, end_train_start_test, end_train_start_test, end_test))
        self.train_start_date_string = start_train
        self.train_end_test_start_date_string = end_train_start_test
        self.test_end_date_string = end_test
        self.preprocess_data()

    def update_indicator(self, list):
        self.input_tech_indicators_list = list
        self.preprocess_data()

    # create a differenced series
    def difference(self, pd):
        return pd.diff().dropna()

    # convert time series into supervised learning problem
    def timeseries_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
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

    def drop_irrelevant_y_var(self, pd, price_exist):
        columns_to_drop = list()
        for i in range(len(self.input_tech_indicators_list)):
            columns_to_drop.append(self.input_tech_indicators_list[i].upper() + "(t)")

        for i in range(len(self.input_tech_indicators_list)):
            for j in range(1, self.n_seq):
                columns_to_drop.append(self.input_tech_indicators_list[i].upper() + "(t+%d)" % j)

        if not price_exist:
            for i in range(self.n_lag):
                columns_to_drop.append("Share Price(t-%d)" % (i+1))

        #print("columns_to_drop", columns_to_drop)
        return pd.drop(columns_to_drop, axis=1)

    # scale train and test data to [-1, 1]
    def scale(self, train_raw, test_raw):
        # fit scaler with 1 Dimensional array data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # display("fit scaler with train data", scaler_train_data)
        scaler = scaler.fit(train_raw)
        # transform train
        train_scaled = scaler.transform(train_raw)
        # display("train_scaled", train_scaled)
        # transform test
        test_scaled = scaler.transform(test_raw)
        # display("test_scaled", test_scaled)

        return scaler, train_scaled, test_scaled

    # fit the model
    def train(self):
        print("Fitting the model")
        start_time = time()
        self.lstm_model = self.fit_lstm(self.train_scaled)
        self.reset()
        self.time_taken_to_train = (time() - start_time)/60
        print("Finished fitting the model, time taken to train: %.1f mins" % self.time_taken_to_train)
        print("Saving object and model")
        self.save()

    def reset(self):
        # forecast the entire training dataset to build up state for forecasting
        # reshape training into [samples, timesteps, features]
        print("Reseting the lstm model")
        self.lstm_model.reset_states()
        total_indicators = len(self.input_tech_indicators_list) + 1
        X, y = self.train_scaled[:, 0:self.n_lag * (len(self.input_tech_indicators_list) + 1)], \
               self.train_scaled[:, self.n_lag * (len(self.input_tech_indicators_list) + 1):]

        if self.model_type == "vanilla" or self.model_type == "stacked" or self.model_type == "bi":
            X = X.reshape(X.shape[0], self.n_lag, total_indicators)
        elif self.model_type == "cnn":
            X = X.reshape(X.shape[0], 1, self.n_lag, total_indicators)
        elif self.model_type == "conv":
            X = X.reshape(X.shape[0], 1, 1, self.n_lag, total_indicators)
        else:
            raise ValueError("self.model_type is not any of the specified")
        self.lstm_model.predict(X, batch_size=1)

    # fit an LSTM network to training data
    def fit_lstm(self, train):
        # reshape training into [samples, timesteps, features]
        # timestesp is 1 as there is 1 sample per day
        X, y = train[:, 0:self.n_lag * (len(self.input_tech_indicators_list) + 1)], \
               train[:, self.n_lag * (len(self.input_tech_indicators_list) + 1):]
        total_indicators = len(self.input_tech_indicators_list) + 1
        # design network
        model = Sequential()
        # source https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        if self.model_type == "vanilla":
            X = X.reshape(X.shape[0], self.n_lag, total_indicators)
            model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, self.n_lag, total_indicators), stateful=True)) #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
            model.add(Dense(y.shape[1]))
        elif self.model_type == "stacked":
            # 2 hidden layers, but can be modified
            X = X.reshape(X.shape[0], self.n_lag, total_indicators)
            model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, self.n_lag, total_indicators), #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
                           return_sequences=True, stateful=True))
            model.add(LSTM(self.n_neurons))
            model.add(Dense(y.shape[1]))
        elif self.model_type == "bi":
            X = X.reshape(X.shape[0], self.n_lag, total_indicators)
            model.add(Bidirectional(LSTM(self.n_neurons, stateful=True),batch_input_shape=(self.n_batch, self.n_lag, total_indicators))) #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
            model.add(Dense(y.shape[1]))

        elif self.model_type == "cnn":
            X = X.reshape(X.shape[0], 1, self.n_lag, total_indicators)
            model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1),
                                      input_shape=(None, self.n_lag, total_indicators))) #batch_input_shape=(None, X.shape[1], X.shape[2])
            model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            model.add(TimeDistributed(Flatten()))
            model.add(LSTM(self.n_neurons))
            model.add(Dense(y.shape[1]))

        elif self.model_type == "conv":
            X = X.reshape(X.shape[0], 1, 1, self.n_lag, total_indicators)
            model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2),
                                 input_shape=(1, 1, self.n_lag, total_indicators)))
            model.add(Flatten())
            model.add(Dense(y.shape[1]))
        else:
            raise ValueError("self.model_type is not any of the specified")

        model.compile(loss='mean_squared_error', optimizer='adam')

        print("train X size", len(X), " train X data dimension", X.shape, "train y size", len(y), " train X data dimension", y.shape)
        #print("train X data", X)
        #print("train y data", y)
        # fit network
        print("Training model with batch size", self.n_batch)
        model.summary()
        for i in range(self.n_epochs):
            model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
            model.reset_states()


        # source https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
        # Create a new model with batch size 1 and give the trained weight, this allows the model
        # to be used to predict 1 step instead of batches
        n_batch = 1
        new_model = Sequential()
        # source https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        if self.model_type == "vanilla":
            new_model.add(LSTM(self.n_neurons, batch_input_shape=(n_batch, self.n_lag, total_indicators), stateful=True)) #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
            new_model.add(Dense(y.shape[1]))
        elif self.model_type == "stacked":
            # 2 hidden layers, but can be modified
            new_model.add(LSTM(self.n_neurons, batch_input_shape=(n_batch, self.n_lag, total_indicators), #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
                           return_sequences=True, stateful=True))
            new_model.add(LSTM(self.n_neurons))
            new_model.add(Dense(y.shape[1]))
        elif self.model_type == "bi":
            new_model.add(Bidirectional(LSTM(self.n_neurons, stateful=True),batch_input_shape=(n_batch, self.n_lag, total_indicators))) #batch_input_shape=(self.n_batch, X.shape[1], X.shape[2])
            new_model.add(Dense(y.shape[1]))

        elif self.model_type == "cnn":
            new_model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1),
                                      input_shape=(None, self.n_lag, total_indicators))) #batch_input_shape=(None, X.shape[1], X.shape[2])
            new_model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            new_model.add(TimeDistributed(Flatten()))
            new_model.add(LSTM(self.n_neurons))
            new_model.add(Dense(y.shape[1]))

        elif self.model_type == "conv":
            new_model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2),
                                 input_shape=(1, 1, self.n_lag, total_indicators)))
            new_model.add(Flatten())
            new_model.add(Dense(y.shape[1]))
        else:
            raise ValueError("self.model_type is not any of the specified")

        new_model.set_weights(model.get_weights())

        print("\n\nNew model with batch size 1 for prediction")
        new_model.summary()
        return new_model

    # make one forecast with an LSTM,
    def forecast_lstm(self, X):
        # make forecast
        total_indicators = len(self.input_tech_indicators_list) + 1
        if self.model_type == "vanilla" or self.model_type == "stacked" or self.model_type == "bi":
            X = X.reshape(1, self.n_lag, total_indicators)
        elif self.model_type == "cnn":
            X = X.reshape(1, 1, self.n_lag, total_indicators)
        elif self.model_type == "conv":
            X = X.reshape(1, 1, 1, self.n_lag, total_indicators)
        else:
            raise ValueError("self.model_type is not any of the specified")
        forecast = self.lstm_model.predict(X, batch_size=1)
        # display("forecast", forecast)
        # convert to array
        return [x for x in forecast[0, :]]

    # make a one-step forecast standalone
    def forecast_lstm_one_step(self):
        self.reset()
        predictions = pd.Series()
        train_index = self.train_raw_series.index

        X = np.array(self.train_scaled[len(self.train_scaled) - 1, self.n_seq:])
        #print("X: ", X, "y: ?")
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
        #print("index", test_index)

        for i in range(len(self.test_scaled)):
            # make multi-step forecast
            X, y = self.test_scaled[i, 0:self.n_lag * (len(self.input_tech_indicators_list) + 1)], \
                   self.test_scaled[i, self.n_lag * (len(self.input_tech_indicators_list) + 1):]
            pred = self.forecast_lstm(X)
            #print("X: ", X, "y: ", y, " pred: ", pred)

            # store forecast
            predictions.at[test_index[i]] = pred
        #display("Len(predictions)", len(predictions), predictions)

        # display("predictions before inverse transform", predictions)
        # inverse transform
        predictions = self.inverse_transform(self.train_raw_series.append(self.test_raw_series), predictions,
                                             len(self.test_raw_series))
        #print("Predictions after inverse transform")
        return predictions

    # evaluate the RMSE for each forecast time step
    def score(self, metric, predictions):
        # convert actual tests and predictions to an appropriate list or arrays
        # construct list of rows
        test_values = self.test_raw_series.values
        actual = list()
        for i in range(len(test_values) - self.n_seq + 1):
            next_days_values = test_values[i: i + self.n_seq]
            actual.append(next_days_values)
        actual = np.array(actual)

        #print("actual", actual)
        #display("Len(actual)", len(actual), actual)

        # excludes the ones that do not have test data
        if self.n_seq == 1:
            predictions = np.array(predictions.tolist())
        else:
            predictions = np.array(predictions.tolist())[:-self.n_seq + 1]

        #display("Len(predictions)", len(predictions), predictions)

        if metric == "rmse":
            rmses = list()
            for i in range(self.n_seq):
                rmse = math.sqrt(mean_squared_error(actual[:, i], predictions[:, i]))
                #print('t+%d RMSE: %f' % ((i + 1), rmse))
                rmses.append(rmse)
            return rmses

        elif metric == "trend":
            # first case is special case since the last data input from the training data is used
            price_1_day_before = self.train_raw_series[-1]
            index = self.test_raw_series.index
            trends = list()
            for i in range(self.n_seq):
                #print("\nCalculating trend score for ", i + 1)
                correct_counts = 0
                for j in range(len(predictions) - i):
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
                    #print("Price 1 day before: ", price_1_day_before)
                    #print("Actual price: ", actual, " | Predicted price: ", predictions[j, i])
                    #print("Actual trend: ", true_trend, " | Predicted trend: ", predicted_trend)
                    # next day
                    price_1_day_before = actual
                price_1_day_before = self.test_raw_series[index[i]]
                print("Correct counts: ", correct_counts, "  Size of test set:", len(self.test_raw_series))
                trends.append(correct_counts / len(self.test_raw_series))
            return trends

        elif metric == "apre":
            apres = list()
            for i in range(self.n_seq):
                apre = np.mean(abs(actual[:, i] - predictions[:, i]) / actual[:, i])
                print('t+%d APRE: %f' % ((i + 1), apre))
                apres.append(apre)
            return apres
        else:
            print(metric, " is not an valid metric. Return NONE")
            return None

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = list()
        new = forecast[0] + last_ob
        inverted.append(new)
        #print("Inverse difference Pred: ", forecast[0], "  + Reference Price:", last_ob, " = ", new)
        last_ob = new
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            new = forecast[i] + inverted[i - 1]
            inverted.append(new)
            #print("Inverse difference Pred: ", forecast[i], "  + Reference Price:", last_ob, " = ", new)
            last_ob = new

        #print("Final inverted values: ", inverted)
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, series, predictions, n_test):
        # walk-forward validation on the test data
        inverted_predictions = pd.Series()
        pred_index = predictions.index
        for i in range(len(predictions)):
            # create array from forecast
            pred = array([0 for i in range(self.n_lag * (len(self.input_tech_indicators_list) + 1))] + predictions[i])
            pred = pred.reshape(1, len(pred))
            # display("pred with place holders", pred)
            # invert scaling
            inv_scale = self.scaler.inverse_transform(pred)[0, self.n_lag * (len(self.input_tech_indicators_list) + 1):]
            # inv_scale = inv_scale[0, :]
            #print("Inverse scale  Original Pred: ", pred, "   After Scaling: ", inv_scale)
                        # invert differencing
            # -1 to get the t-1 price
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted_predictions.at[pred_index[i]] = inv_diff
        return inverted_predictions

    def save_lstm_model(self):
        self.lstm_model.save("models/" + self.create_file_name() + ".h5")

    def save_object(self):
        temp_model = self.lstm_model
        temp_holiday = self.us_holidays
        # not pickable so reload these at a later stage
        self.lstm_model = None
        self.us_holidays = None
        # don't save the Keras model
        with open("objects/" + self.create_file_name() + ".pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        self.lstm_model = temp_model
        self.us_holidays = temp_holiday

    def create_file_name(self):
        file_name = self.name + "_" + self.model_type + "_nlag" + str(self.n_lag) \
                    + "_nseq" + str(self.n_seq) + "_e" + str(self.n_epochs)\
                    + "_b" + str(self.n_batch) + "_n" + str(self.n_neurons) \
                    + "_ind" + str(len(self.input_tech_indicators_list)+1) \
                    + "_train" + self.train_start_date_string \
                    + "_trainendteststart" + self.train_end_test_start_date_string \
                    + "_testend" + self.test_end_date_string
        file_name = file_name.replace("/", "-")
        return file_name

    def save(self):
        self.save_lstm_model()
        self.save_object()

    def save_raw_pd_to_csv(self):
        self.raw_pd.to_csv("raw_data/" + self.name + "_raw_pd.csv")

    # plot function for children classes, if run by parent, error would happen
    def plot(self, predictions, start_date_string=None, end_date_string=None):
        # line plot of observed vs predicted
        formatter = matplotlib.dates.DateFormatter('%d/%m/%Y')
        if start_date_string is not None and start_date_string is not None:
            predictions = self.get_filtered_series(predictions, start_date_string, end_date_string)
            train_raw_series = self.get_filtered_series(self.train_raw_series, start_date_string, end_date_string)
            test_raw_series = self.get_filtered_series(self.test_raw_series, start_date_string, end_date_string)

        else:
            train_raw_series = self.train_raw_series
            test_raw_series = self.test_raw_series
        # display(predictions.index)
        for test_date in predictions.index:
            # n_seq consecutive days
            x_axis = list()
            # The first day of test
            x_axis.append(test_date)
            for j in range(self.n_seq - 1):  # first one already added
                x_axis.append(self.date_by_adding_business_days(from_date=x_axis[-1], add_days=1))
            plt.plot(x_axis, predictions[test_date], ':', marker='*', color="blue", label="Predicted prices")

        plt.plot(train_raw_series.index, train_raw_series.values,
                 '-', marker=".", label="Actual prices (Training data)")

        # test data not always possible
        try:
            plt.plot(test_raw_series.index, test_raw_series.values,
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
