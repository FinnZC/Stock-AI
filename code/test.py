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
            self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string, start_delay=1)
            self.train_scaled, self.test_scaled = self.preprocess_data()

        # create a differenced series
        def difference(self, series, source, interval=1):
            diff = list()
            # First item is special case because we use the difference of the last training pair to predict the first test price
            if source == "test":
                diff.append(self.train_raw_series.values[-1] - self.train_raw_series.values[-2])

            for i in range(1, len(series)):
                value = series[i] - series[i-1]
                diff.append(value)

            # Last item is special case because there is no next value thus the diff is
            # 1 size shorter than the original test_raw. We fix this by adding an additional item
            if source == "test":
                diff.append(0) # placeholder for the last prediction, not used in anyway

            return pd.Series(diff)

        # adapted from https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
        def timeseries_to_supervised(self, data, lag=1):
            df = pd.DataFrame(data)
            columns = [df.shift(i) for i in range(1, lag+1)]
            columns.append(df)
            df = pd.concat(columns, axis=1)
            df.fillna(0, inplace=True)
            return df

        # invert differenced value
        def inverse_difference(self, history, yhat, interval=1):
            #print("interval", interval)
            #display(history)
            return yhat + history.values[-interval]

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

        # inverse scaling for a forecasted value
        def invert_scale(self, X, value):
            new_row = [x for x in X] + [value]
            array = numpy.array(new_row)
            array = array.reshape(1, len(array))
            inverted = self.scaler.inverse_transform(array)
            return inverted[0, -1]

        def preprocess_data(self):
            # transform data to be stationary
            train_diff_values = self.difference(self.train_raw_series.values, "train", 1)
            test_diff_values = self.difference(self.test_raw_series.values, "test", 1)

            # transform data to be supervised learning
            train_supervised_pd = self.timeseries_to_supervised(train_diff_values, 1)
            train = train_supervised_pd.values

            # removes first row because it is not relevant
            test_supervised_pd = self.timeseries_to_supervised(test_diff_values, 1).iloc[1:]
            test = test_supervised_pd.values

            # transform the scale of the data
            scaler, train_scaled, test_scaled = self.scale(train, test)
            self.scaler = scaler

            """
            print("size of train_raw data: ", len(self.train_raw_series))
            display(self.train_raw_series)

            print("size of diff train data: ", len(train_diff_values))
            display(train_diff_values)
            print("size of supervised train data: ", len(train))
            display(train)


            print("size of test_raw data: ", len(self.test_raw_series))
            display(self.test_raw_series)
            print("size of diff test data: ", len(test_diff_values))
            display(test_diff_values)
            print("size of supervised test data: ", len(test))
            display(test)
            """
            print("size of supervised train_scaled data: ", len(train_scaled))
            display(train_scaled)
            print("size of supervised test_scaled data: ", len(test_scaled))
            display(test_scaled)

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
            X, y = train[:, 0:-1], train[:, -1]
            X = X.reshape(X.shape[0], 1, X.shape[1])
            model = Sequential()
            model.add(LSTM(self.n_neurons, batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]), stateful=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            for i in range(self.n_epoch):
                model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
                model.reset_states()
            return model

        # make a one-step forecast within test
        def forecast_lstm(self, X):
            X = X.reshape(1, 1, len(X))
            pred = self.lstm_model.predict(X, batch_size=self.n_batch)
            return pred[0,0]

        # make a one-step forecast standalone
        def forecast_lstm_one_step(self):
            predictions = pd.Series()
            unknown_next_day_price = self.scaler.transform([[0, 0]])

            print("Next day: ", unknown_next_day_price)
            X, y = unknown_next_day_price[0, 0:-1], unknown_next_day_price[0, -1]
            X = X.reshape(1, 1, len(X))
            print("X: ", X, "y: ", y)
            # predicting the change
            pred_price = self.lstm_model.predict(X, batch_size=self.n_batch)
            # invert scaling
            pred_price = self.invert_scale(X, pred_price)
            # invert differencing
            pred_price = pred_price + self.train_raw_series.values[-1]

            # Prediction for the next business working day
            predictions.at[self.date_by_adding_business_days(
                from_date=self.train_raw_series.index[-1], add_days=1)] = pred_price
            display(predictions)
            return predictions

        def predict(self):
            # walk-forward validation on the test data
            predictions = pd.Series()
            # Index is datetime
            test_index = self.test_raw_series.index
            #predict the fist share price after the last share price in the training data
            #pred = self.forecast_lstm(1, self.train_scaled[i, 0:-1])
            for i in range(len(self.test_scaled)):
                # make one-step forecast
                X, y = self.test_scaled[i, 0:-1], self.test_scaled[i, -1]
                print("X: ", X, "y: ", y)
                pred = self.forecast_lstm(X)
                # invert scaling
                pred = self.invert_scale(X, pred)
                # invert differencing
                pred = self.inverse_difference(self.test_raw_series, pred, len(self.test_scaled)-i)
                # store forecast
                predictions.at[test_index[i]] = pred

                #expected = self.invert_scale(X, y)
                #expected = self.inverse_difference(self.test_raw_series, expected, len(self.test_scaled)-i)
                #exp = self.test_raw_series[test_index[i]]
                #print('Predicted=%f, Expected Raw = %f' % (pred, exp))

            display("predictions", predictions)
            return predictions
