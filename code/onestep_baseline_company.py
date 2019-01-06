import pandas as pd
from company import Company

class OneStepBaselineCompany(Company):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string):
        Company.__init__(self, name)
        self.train_raw_series = self.get_share_prices(train_start_date_string, train_end_test_start_date_string)
        self.test_raw_series = self.get_share_prices(train_end_test_start_date_string, test_end_date_string,
                                                     start_delay=1)

    def train(self):
        pass

    def predict(self):
        predictions = pd.Series()
        # Persistence Model Forecast, basically, the same share price with 1 date timelag
        # e.g. the predicted share price at time t, is t-1
        if len(self.test_raw_series) > 0:
            predictions = self.test_raw_series.shift(1)
            # Special case of the first value is changed to the zero
            predictions.at[predictions.index[0]] = self.train_raw_series.values[-1]
        else:
            # no test data inputed
            # the predicted next day price is the last price of the training data
            predictions.at[
                self.date_by_adding_business_days(
                    self.train_raw_series.index[-1], 1)] = self.train_raw_series.values[-1]

        return predictions

