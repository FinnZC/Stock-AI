from multistep_lstm_company import MultiStepLSTMCompany
from alpha_vantage.techindicators import TechIndicators
from time import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error



# No preprocessing in this stage, see if it is good
class MultiStepLSTMCompanyTechIndicatorsWithDifferencing(MultiStepLSTMCompany):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons, tech_indicators=[]):
        self.tech_indicators = TechIndicators(key='3OMS720IM6CRC3SV', output_format='pandas')
        self.all_tech_indicators = ["ad", "adosc", "adx", "adxr", "apo", "aroon", "aroonosc",
                                    "bbands", "bop", "cci", "cmo", "dema", "dx", "ema", "ht_dcperiod",
                                    "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode",
                                    "kama", "macd", "macdext", "mama", "mfi", "midpoint", "midprice",
                                    "minus_di", "minus_dm", "mom", "natr", "obv", "plus_di", "plus_dm",
                                    "ppo", "roc", "rocr", "rsi", "sar", "sma", "stoch", "stochf", "stochrsi",
                                    "t3", "tema", "trange", "trima", "trix", "ultsoc", "willr", "wma"]
        if tech_indicators == "all":
            self.input_tech_indicators_list = self.all_tech_indicators
        else:
            self.input_tech_indicators_list = tech_indicators
        self.n_indicators = len(tech_indicators)
        self.supervised_pd = None
        # self.all_tech_indicators
        MultiStepLSTMCompany.__init__(self, name, train_start_date_string, train_end_test_start_date_string,
                                      test_end_date_string,
                                      n_lag, n_seq, n_epochs, n_batch, n_neurons)

    def add_tech_indicators_dataframe(self, price_series, indicators):
        combined = price_series
        for ind in indicators:
            print("ind", ind)
            while True:  # try again until success
                try:
                    ind_series = self.get_indicator(ind, self.train_start_date_string, self.test_end_date_string)
                    combined = pd.concat([combined, ind_series], axis=1)
                    break
                except:
                    print("Retrying to download indicator ", ind)
                    pass

        return combined

    def get_indicator(self, ind_name, start, end):
        data, meta_data = getattr(self.tech_indicators, "get_" + ind_name)(self.name, interval="daily")
        data.index = pd.to_datetime(data.index)
        return data

    # create a differenced series
    def difference(self, pd):
        return pd.diff().dropna()

    def preprocess_data(self):
        price_series = self.share_prices_series
        display("price data series", len(price_series), price_series)
        if len(self.input_tech_indicators_list) > 0 and self.supervised_pd is None:
            # add additional technical indicators
            combined = self.add_tech_indicators_dataframe(price_series, self.input_tech_indicators_list)
        else:
            combined = price_series

        # display("combined", combined)
        if self.supervised_pd is None:
            supervised_pd = self.timeseries_to_supervised(combined, self.n_lag, self.n_seq)
            # display("supervised", supervised_pd)
            # delete unnecessary variables for prediction except price (should be var1)
            supervised_pd = self.drop_irrelevant_y_var(supervised_pd)
            display("supervised_pd original", supervised_pd)
            supervised_pd = self.difference(supervised_pd)
            display("supervised_pd after differencing", supervised_pd)
            self.supervised_pd = supervised_pd

        supervised_pd = self.get_filtered_series(self.supervised_pd, self.train_start_date_string, self.test_end_date_string)
        cutoff = len(self.train_raw_series) - self.n_seq
        train_supervised_values = supervised_pd.values[:cutoff - self.n_lag + 1]
        # display("train supervised values", train_supervised_values)
        test_supervised_values = supervised_pd.values[cutoff + self.n_seq - self.n_lag:]
        # display("test supervised values", test_supervised_values)

        display("filtered train values", supervised_pd)

        self.scaler, scaled_train_supervised, scaled_test_supervised = self.scale(train_supervised_values,
                                                                                  test_supervised_values)
        display("scaled train supervised", scaled_train_supervised)
        display("scaled test supervised", scaled_test_supervised)

        self.train_scaled, self.test_scaled = scaled_train_supervised, scaled_test_supervised

    def drop_irrelevant_y_var(self, pd):
        columns_to_drop = list()
        for i in range(self.n_indicators):
            columns_to_drop.append(self.input_tech_indicators_list[i].upper() + "(t)")

        for i in range(self.n_indicators):
            for j in range(1, self.n_seq):
                columns_to_drop.append(self.input_tech_indicators_list[i].upper() + "(t+%d)" % j)
        return pd.drop(columns_to_drop, axis=1)

    # evaluate the persistence model
    def predict(self):
        self.reset()
        # walk-forward validation on the test data
        predictions = pd.Series()
        # Index is datetime
        test_index = self.test_raw_series.index
        # print("index", test_index)
        for i in range(len(self.test_scaled)):
            # make multi-step forecast
            X, y = self.test_scaled[i, 0:self.n_lag * (self.n_indicators + 1)], self.test_scaled[i,
                                                                                self.n_lag * (self.n_indicators + 1):]
            print("X: ", X, "y: ", y)
            pred = self.forecast_lstm(X)
            print("Prediction: ", pred)
            # store forecast
            # print(test_index[i])
            predictions.at[test_index[i]] = pred
            # display(predictions)

        # display("predictions before inverse transform", predictions)
        # inverse transform
        predictions = self.inverse_transform(self.train_raw_series.append(self.test_raw_series), predictions,
                                             len(self.test_raw_series))
        print("Predictions after inverse transform")
        display(predictions)
        return predictions

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

    # fit an LSTM network to training data
    def fit_lstm(self, train):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:self.n_lag * (self.n_indicators + 1)], train[:, self.n_lag * (self.n_indicators + 1):]
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
        # display("forecast", forecast)
        # convert to array
        return [x for x in forecast[0, :]]

    def reset(self):
        # forecast the entire training dataset to build up state for forecasting
        # reshape training into [samples, timesteps, features]
        self.lstm_model.reset_states()
        X, y = self.train_scaled[:, 0:self.n_lag * (self.n_indicators + 1)], self.train_scaled[:,
                                                                             self.n_lag * (self.n_indicators + 1):]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.lstm_model.predict(X, batch_size=self.n_batch)

    # inverse data transform on forecasts
    def inverse_transform(self, series, predictions, n_test):
        # walk-forward validation on the test data
        inverted_predictions = pd.Series()
        pred_index = predictions.index
        for i in range(len(predictions)):
            # create array from forecast
            pred = array([0 for i in range(self.n_lag * (self.n_indicators + 1))] + predictions[i])
            pred = pred.reshape(1, len(pred))
            # display("pred with place holders", pred)
            # invert scaling
            inv_scale = self.scaler.inverse_transform(pred)[0, self.n_lag * (self.n_indicators + 1):]
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

    # evaluate the RMSE for each forecast time step
    def score(self, metric, predictions):
        # convert actual tests and predictions to an appropriate list or arrays
        # construct list of rows
        # first item is test data for the next days, hence not taken into account to measure the prediction
        test_values = self.test_raw_series.values[:-self.n_seq+1 if self.n_seq > 1 else len(self.test_raw_series)]
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