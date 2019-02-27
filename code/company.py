import pandas as pd
import datetime
import holidays
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from time import sleep


class Company(object):
    def __init__(self, name):
        self.name = name
        # Create object to request data from Alpha Vantage
        self.time_series = TimeSeries(key='3OMS720IM6CRC3SV', output_format='pandas', indexing_type='date')
        self.tech_indicators = TechIndicators(key='3OMS720IM6CRC3SV', output_format='pandas')
        self.all_tech_indicators = ["ad", "adosc", "adx", "adxr", "apo", "aroon", "aroonosc",
                                    "bbands", "bop", "cci", "cmo", "dema", "dx", "ema", "ht_dcperiod",
                                    "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode",
                                    "kama", "macd", "macdext", "mama", "mfi", "midpoint", "midprice",
                                    "minus_di", "minus_dm", "mom", "natr", "obv", "plus_di", "plus_dm",
                                    "ppo", "roc", "rocr", "rsi", "sar", "sma", "stoch", "stochf", "stochrsi",
                                    "t3", "tema", "trange", "trima", "trix", "ultsoc", "willr", "wma"]
        while True:
            try:
                price_series, metadata = self.time_series.get_daily_adjusted(symbol=self.name, outputsize='full')
                break
            except KeyError:
                # Could be that API has reached its limit
                print("Retrying to download price series")
                sleep(20)
                pass
        # Convert index of the DataFrame which is in the date string format into datetime
        price_series.index = pd.to_datetime(price_series.index)
        self.share_prices_series = price_series["5. adjusted close"]  # Series
        self.share_prices_series.name = "Share Price"
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
    
    # Get share prices or technical indidcators within the range
    def get_filtered_series(self, series, start_date_string=None, end_date_string=None, start_delay=None):
        # Check whether there needs "days" delay in the returned share prices
        if start_delay != None:
            start_date = self.date_by_adding_business_days(
                from_date=self.convert_date_string_to_datetime(start_date_string), add_days=start_delay)
        else:
            start_date = self.convert_date_string_to_datetime(start_date_string)
        end_date = self.convert_date_string_to_datetime(end_date_string)
        relevant_series_range = series[
            (series.index >= start_date) & (series.index <= end_date)]
        return relevant_series_range

