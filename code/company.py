import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import math
import holidays
from sklearn.metrics import mean_squared_error
from alpha_vantage.timeseries import TimeSeries


class Company(object):
    def __init__(self, name):
        self.name = name
        # Create object to request data from Alpha Vantage
        self.time_series = TimeSeries(key='3OMS720IM6CRC3SV', output_format='pandas', indexing_type='date')
        data, metadata = self.time_series.get_daily_adjusted(symbol=name, outputsize='full')
        # Convert index of the DataFrame which is in the date string format into datetime
        data.index = pd.to_datetime(data.index)
        self.converted_dates = data.index  # DateTimeIndex64
        self.share_prices_series = data["5. adjusted close"]  # Series
        self.us_holidays = holidays.UnitedStates()
    
    def convert_date_string_to_datetime(self, date_string):
        date_day, date_month, date_year = date_string.split("/")
        return datetime.datetime(int(date_year), int(date_month), int(date_day), 0, 0)
    
    # add business days (excluding weekends only, but does not take into account
    # HOLIDAYS as they vary country by country)
    def date_by_adding_business_days(self, from_date, add_days):
        business_days_to_add = add_days
        current_date = from_date
        while business_days_to_add > 0:
            current_date += datetime.timedelta(days=1)
            weekday = current_date.weekday()
            if weekday >= 5 or current_date in self.us_holidays: # sunday = 6
                continue
            business_days_to_add -= 1
        return current_date
    
    # Get share prices within the range
    def get_share_prices(self, start_date_string=None, end_date_string=None, start_delay=None):
        # Check whether there needs "days" delay in the returned share prices
        if start_delay != None:
            start_date = self.date_by_adding_business_days(
                from_date=self.convert_date_string_to_datetime(start_date_string), add_days=start_delay)
        else:
            start_date = self.convert_date_string_to_datetime(start_date_string)
        end_date = self.convert_date_string_to_datetime(end_date_string)
        relevant_share_prices = self.share_prices_series[
            (self.share_prices_series.index >= start_date) & (self.share_prices_series.index <= end_date)]
        return relevant_share_prices
        
    # plot function for children classes, if run by parent, error would happen
    def plot(self, predictions):
        # line plot of observed vs predicted
        formatter = matplotlib.dates.DateFormatter('%d/%m/%Y')     
        
        plt.plot(predictions.index, predictions.values, ':', marker='*', label="Predicted prices", color="blue")
        plt.plot(self.train_raw_series.index, self.train_raw_series.values,
                 '-', marker= ".", label="Actual prices (Training data)")
            
        # test data not always possible
        try:
            plt.plot(self.test_raw_series.index[:len(predictions)], 
                self.test_raw_series.values[:len(predictions)], '-', marker=".", label="Actual prices (Test data)")
        except:
            # don't plot test data if not available
            print("Exception entered")
            pass

        ax = plt.gcf().axes[0]
        ax.xaxis.set_major_formatter(formatter)
        ax.legend()
        plt.gcf().autofmt_xdate(rotation=25)
        plt.gcf().set_size_inches(15, 10)
        plt.xlabel("Time")
        plt.ylabel("Share Price ($)")
        plt.title("Stock price prediction for " + self.name)
        plt.show()
        
    # score function for children classes, if run by parent, error would happen
    def score(self, metric, predictions):
        if self.test_raw_series.empty:
            raise ValueError("No test data passed so unable to score")
        elif len(self.test_raw_series) != len(predictions):
            raise ValueError("Len of test data is not equal the length of predicted data")
        
        # predictions and self.test_raw_series are series with index representing its original index in the dataset
        # root mean squared error
        if metric == "rmse":
            rmse = math.sqrt(mean_squared_error(self.test_raw_series, predictions))
            return rmse
        # trend whether the prediction for the next day is up or down and its accuracy
        elif metric == "trend":
            # first case is special case since the last data input from the training data is used
            price_1_day_before = self.train_raw_series[self.train_raw_series.index.tolist()[-1]]
            correct_counts = 0
            index = self.test_raw_series.index
            for i in range(1, len(self.test_raw_series)):
                if self.test_raw_series[index[i]] > price_1_day_before:
                    true_trend = "up"
                elif self.test_raw_series[index[i]] < price_1_day_before:
                    true_trend = "down"
                else:
                    true_trend = "neutral"
                
                if predictions[index[i]] > price_1_day_before:
                    predicted_trend = "up"
                elif predictions[index[i]] < price_1_day_before:
                    predicted_trend = "down"
                else:
                    predicted_trend = "neutral"
                
                if true_trend == predicted_trend:
                    correct_counts += 1
                #print("Price 1 day before", price_1_day_before)
                #print("Actual price: ", self.test_raw_series[index[i]], " | Predicted price: ", predictions[index[i]])
                #print("Actual trend: ", true_trend, " | Predicted trend: ", predicted_trend)
                # next day
                price_1_day_before = self.test_raw_series[index[i]]
            
            return correct_counts/len(self.test_raw_series)