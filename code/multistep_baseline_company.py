import numpy as np
import pandas as pd
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from company import Company
from multistep_lstm_company import MultiStepLSTMCompany


# Starts multi-step forecasting
class MultiStepBaselineCompany(MultiStepLSTMCompany):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq,  n_batch=None, tech_indicators=[], model_type="vanilla"):
        MultiStepLSTMCompany.__init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq,  n_batch, tech_indicators, model_type)

    # make a persistence forecast
    def persistence(self, last_ob):
        return [last_ob for i in range(self.n_seq)]

    # evaluate the persistence model
    def predict(self):
        predictions = pd.Series()
        # Index is datetime
        test_index = self.test_raw_series.index
        share_prices = self.train_raw_series.append(self.test_raw_series).values
        #display(self.train_raw_series)
        #display(self.test_raw_series)

        for i in range(len(test_index)):
            # print("X: ", X, "y: ", y)
            # make forecast
            pred = self.persistence(share_prices[len(self.train_raw_series)+i-1])
            # store the forecast
            predictions.at[test_index[i]] = pred
            # predictions.append(pred)
        #display(predictions)
        return predictions
