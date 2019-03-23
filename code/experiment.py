import numpy as np
import csv
import os
from multistep_lstm_company import MultiStepLSTMCompany
from datetime import date
import math
import pandas as pd


def experiment(file_output_name, symbol, start_train_date, end_train_start_test_date, end_test_date, n_lags,
                                            n_seqs, n_batches, indicators, model_types):
    # This is optimising parameters for n_epochs, n_batch, and n_neurons
    # param = {"n_epochs": n_epochs, "n_batch": n_batch, "n_neurons": n_neurons}
    csv_columns = ["Company", "LSTM Type", "n_epoch", "n_batch",
                   "n_lag", "n_seq", "Training Time",
                   "Indicator Number", "Indicators", "Trained Date",
                   "Start Train Date", "End Train/Start Test Date", "End Test Date",
                   "Model Name"]
    for i in range(30):
        csv_columns.append("Trend_t+" + str(i + 1))
        csv_columns.append("APRE_t+" + str(i + 1))
        csv_columns.append("RMSE_t+" + str(i + 1))

    if not os.path.isdir("./experiments"):
        # create directory
        os.mkdir("experiments")

    filename = "./experiments/" + file_output_name + ".csv"
    if not os.path.isfile(filename):
        # create new file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    for n_b in n_batches:
        for n_l in n_lags:
            for n_s in n_seqs:
                for m_t in model_types:
                    for ind in indicators:
                        print("In process of training", symbol, "model type:", m_t, "n_lag:",
                                 n_l, "n_seq:", n_s, "n_batch:", n_b,)
                        obj = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date,
                                                       end_test_date, n_lag=n_l, n_seq=n_s, n_batch=n_b, tech_indicators=ind,
                                                       model_type=m_t)
                        obj.train()
                        predictions = obj.predict()
                        trend_score = obj.score(metric="trend", predictions=predictions)
                        lstm_score = obj.score(metric="rmse", predictions=predictions)
                        apre_score = obj.score(metric="apre", predictions=predictions)
                        dic = {"Company": symbol,
                                       "LSTM Type": obj.model_type,
                                       "n_epoch": obj.n_epochs,
                                       "n_batch": obj.n_batch,
                                       "n_lag": obj.n_lag,
                                       "n_seq": obj.n_seq,
                                       "Training Time": obj.time_taken_to_train,
                                       "Start Train Date": obj.train_start_date_string,
                                       "End Train/Start Test Date": obj.train_end_test_start_date_string,
                                       "End Test Date": obj.test_end_date_string,
                                       "Indicator Number": len(obj.input_tech_indicators_list),
                                       "Indicators":  ",".join(obj.input_tech_indicators_list),
                                       "Trained Date": str(date.today()),
                                       "Model Name": obj.create_file_name()}
                        for i in range(n_s):
                            dic["Trend_t+" + str(i + 1)] = trend_score[i]
                            dic["APRE_t+" + str(i + 1)] = apre_score[i]
                            dic["RMSE_t+" + str(i + 1)] = lstm_score[i]
                        append_dict_to_csv(filename, csv_columns, dic)


def append_dict_to_csv(csv_file_name, csv_columns, dic):
    with open(csv_file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(dic)




def eperiment_n_epochs():
    n_lags = [3]
    n_seqs = [3]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = [["price"]]
    model_types = ["vanilla", "stacked", "bi", "cnn",
                   "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2017"
    end_test_date = "01/01/2018"

    experiment(file_output_name="optimisation_e_poch_optimisation", symbol="AMZN", start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
               end_test_date=end_test_date, n_lags=n_lags,
               n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)


def experiment_1_univariate():
    n_lags = [3]
    n_seqs = [1]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    all_tech_indicators = ["price", "ad", "adosc", "adx", "adxr", "apo", "aroon", "aroonosc",
                           "bbands", "bop", "cci", "cmo", "dema", "dx", "ema", "ht_dcperiod",
                           "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode",
                           "kama", "macd", "macdext", "mama", "mfi", "midpoint", "midprice",
                           "minus_di", "minus_dm", "mom", "natr", "obv", "plus_di", "plus_dm",
                           "ppo", "roc", "rocr", "rsi", "sar", "sma", "stoch", "stochf", "stochrsi",
                           "t3", "tema", "trange", "trima", "trix", "ultsoc", "willr", "wma"]
    indicators = [[ind] for ind in all_tech_indicators]
    model_types = ["cnn", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"

    experiment(file_output_name="experiment_1_univariate_laptop", symbol="AMZN", start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
               end_test_date=end_test_date, n_lags=n_lags,
               n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def experiment_2_part1():
    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()

    n_lags = [1]
    n_seqs = [1]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = [["ht_trendline"], ["ht_trendline","natr","midpoint","mfi","trix","mama","trima","obv","aroon","stoch","rocr","cci","plus_dm","t3","kama","ema","tema","aroonosc","ultsoc","sma","minus_di","trange","stochrsi","ht_phasor","adosc","bbands","ppo","stochf"]]
    model_types = ["vanilla", "stacked", "bi", "cnn", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"

    for symbol in nasdaq_100_symbols:
        experiment(file_output_name="experiment_2_part1", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def experiment_2_part2():
    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()

    n_lags = [3]
    n_seqs = [1]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = None # TODO: []
    model_types = ["vanilla", "stacked", "bi", "cnn", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"

    for symbol in nasdaq_100_symbols:
        experiment(file_output_name="experiment_2_part2", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def experiment_3():
    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()

    n_lags = [3]
    n_seqs = [1, 3]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = None # TODO
    model_types = ["vanilla", "stacked", "bi", "cnn", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"

    for symbol in nasdaq_100_symbols:
        experiment(file_output_name="experiment_3", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def experiment_4():
    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()

    n_lags = [1, 3]
    n_seqs = [1]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = None # TODO
    model_types = ["vanilla", "stacked", "bi", "cnn", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"

    for symbol in nasdaq_100_symbols:
        experiment(file_output_name="experiment_4", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)
