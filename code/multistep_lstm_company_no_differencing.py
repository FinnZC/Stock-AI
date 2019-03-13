from multistep_lstm_company import MultiStepLSTMCompany
import pandas as pd
from time import time
from time import sleep

class MultiStepLSTMCompanyNoDifferencing(MultiStepLSTMCompany):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_neurons, n_batch=None, tech_indicators=[], model_type="vanilla"):
        MultiStepLSTMCompany.__init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_neurons, n_batch, tech_indicators, model_type)

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

        # display("price data series", len(price_series), price_series)
        if len(self.input_tech_indicators_list) > 0:
            # add additional technical indicators
            combined = self.add_tech_indicators_dataframe(self.share_prices_series, self.input_tech_indicators_list)
        else:
            combined = self.share_prices_series

        self.raw_pd = combined

        # display("train raw series", self.train_raw_series)
        # display("test raw series", self.test_raw_series)

        supervised_pd = self.timeseries_to_supervised(combined, self.n_lag, self.n_seq)
        # display("supervised", supervised_pd)
        # delete unnecessary variables for prediction except price (should be var1)
        supervised_pd = self.drop_irrelevant_y_var(supervised_pd, price_in_ind_list)
        # display("supervised_pd original", supervised_pd)

        supervised_pd = self.get_filtered_series(supervised_pd, self.train_start_date_string, self.test_end_date_string)


        self.supervised_pd = supervised_pd
        # display("supervised filtered pd ", supervised_pd)
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

        # display("test supervised values", test_supervised_values)

        # display("filtered train values", supervised_pd)

        self.scaler, scaled_train_supervised, scaled_test_supervised = self.scale(train_supervised_values,
                                                                                  test_supervised_values)
        # display("scaled train supervised", scaled_train_supervised.shape)
        # display("scaled test supervised", scaled_test_supervised)
        self.train_scaled, self.test_scaled = scaled_train_supervised, scaled_test_supervised
        print("Preprocessed data in ", (time() - start_time ) /60, "mins")

        if self.n_batch == "full_batch":
            self.n_batch = len(self.train_raw_series)
        elif self.n_batch == "online":
            self.n_batch = 1
        elif self.n_batch == "half_batch":
            half = int(len(self.train_raw_series ) /2)
            for i in range(half)[::-1]:
                if len(self.train_raw_series) % i == 0:
                    self.n_batch = i
                    break
        else:
            raise ValueError("n_batch is not full_batch, half_batch, nor online. Must be one of them!")