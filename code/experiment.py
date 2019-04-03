import numpy as np
import csv
import os
from multistep_lstm_company import MultiStepLSTMCompany
from datetime import datetime
import math
import pandas as pd

global n_experiment
global progress
progress = 0


def experiment(file_output_name,symbol, start_train_date, end_train_start_test_date, end_test_date, n_lags,
                                            n_seqs, n_batches, indicators, model_types):
    global n_experiment
    global progress

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
                        if n_l == 1 and m_t == "conv":
                            pass
                        else:
                            print("In process of training", symbol, "model type:", m_t, "n_lag:",
                                  n_l, "n_seq:", n_s, "n_batch:", n_b)
                            if ind == "all":
                                csv_ind = ["ad", "adosc", "adx", "adxr", "apo", "aroon", "aroonosc",
                                           "bbands", "bop", "cci", "cmo", "dema", "dx", "ema", "ht_dcperiod",
                                           "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode",
                                           "kama", "macd", "macdext", "mama", "mfi", "midpoint", "midprice",
                                           "minus_di", "minus_dm", "mom", "natr", "obv", "plus_di", "plus_dm",
                                           "ppo", "roc", "rocr", "rsi", "sar", "sma", "stoch", "stochf", "stochrsi",
                                           "t3", "tema", "trange", "trima", "trix", "ultsoc", "willr", "wma", "price"]
                            else:
                                csv_ind = ind
                            row = {"Company": symbol,
                                   "LSTM Type": m_t,
                                   "n_lag": str(n_l),
                                   "n_seq": str(n_s),
                                   "Indicators": ",".join(csv_ind),
                                   }
                            if not row_exist(filename, row):
                                obj = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date,
                                                           end_test_date, n_lag=n_l, n_seq=n_s, n_batch=n_b,
                                                           tech_indicators=ind,
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
                                       "Indicators": ",".join(obj.input_tech_indicators_list),
                                       "Trained Date": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                                       "Model Name": obj.create_file_name()}
                                for i in range(n_s):
                                    dic["Trend_t+" + str(i + 1)] = trend_score[i]
                                    dic["APRE_t+" + str(i + 1)] = apre_score[i]
                                    dic["RMSE_t+" + str(i + 1)] = lstm_score[i]
                                append_dict_to_csv(filename, csv_columns, dic)
                            progress += 1
                            print("progress:", progress)
                            print("total", n_experiment)
                            print("--------------PROGRESS %.2f %%---------------" % ((progress / n_experiment) * 100))

def append_dict_to_csv(csv_file_name, csv_columns, dic):
    with open(csv_file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(dic)


def row_exist(filename, dic):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                row_dic = {k: row[k] for k in dic.keys() if k in row.keys()}
                if row_dic == dic:
                    print(dic, " exist")
                    return True
            except:
                # skip exceptions which are wrongly formatted rows
                pass
        else:
            print(dic, " does not exist")
            return False

"""
# old experiment
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
"""

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
    indicators = [[ind] for ind in all_tech_indicators][:40]
    model_types = ["cnn"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types)
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
    ranked_ind = ["ht_trendline","natr","mfi","midpoint","trima","trix","mama","obv","aroon","stoch","rocr","cci","plus_dm","t3","kama","ema","tema","aroonosc","ultsoc","sma","minus_di","trange","stochrsi","ht_phasor","adosc","bbands","ppo","stochf","plus_di","rsi","roc","willr","ht_dcperiod","cmo","midprice","adxr","dema","bop","ad","dx","price","mom","macdext","adx","sar","apo","wma","ht_dcphase","ht_sine","minus_dm","macd","ht_trendmode"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=5)).astype(int)]

    model_types = ["bi"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * 103
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
    ranked_ind = ["trix","mama","ad","ppo","trima","adx","minus_di","rsi","obv","natr","minus_dm","aroon","sar","cmo","stochrsi","stochf","wma","midprice","t3","macdext","rocr","ht_dcphase","roc","ht_phasor","ht_dcperiod","ht_sine","dema","aroonosc","sma","bop","apo","adosc","willr","mfi","ultsoc","macd","dx","kama","trange","adxr","bbands","midpoint","ht_trendline","tema","ht_trendmode","stoch","plus_di","cci","plus_dm","ema","mom","price"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=5)).astype(int)]
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * 103
    for symbol in nasdaq_100_symbols:
        experiment(file_output_name="experiment_2_part2", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

# Testing the effect of n lags
def experiment_3():
    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()

    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(10, 10), num=4)).astype(int))
    n_lags.remove(1)
    n_seqs = np.ceil(np.logspace(math.log(1, 10), math.log(10, 10), num=4)).astype(int)

    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    ranked_ind = ["trix","mama","ad","ppo","trima","adx","minus_di","rsi","obv","natr","minus_dm","aroon","sar","cmo","stochrsi","stochf","wma","midprice","t3","macdext","rocr","ht_dcphase","roc","ht_phasor","ht_dcperiod","ht_sine","dema","aroonosc","sma","bop","apo","adosc","willr","mfi","ultsoc","macd","dx","kama","trange","adxr","bbands","midpoint","ht_trendline","tema","ht_trendmode","stoch","plus_di","cci","plus_dm","ema","mom","price"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=4)).astype(int)]
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/2000"
    end_train_start_test_date = "01/01/2018"
    end_test_date = "01/01/2019"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * 103
    for symbol in nasdaq_100_symbols[60:80][::-1]:
        experiment(file_output_name="experiment_3", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                   end_test_date=end_test_date, n_lags=n_lags,
                   n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)


def benchmark_abraham2004modeling_nasdaq():
    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(30, 10), num=15)).astype(int))
    n_seqs = list(np.ceil(np.logspace(math.log(1, 10), math.log(10, 10), num=1)).astype(int))

    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = [["price"]]
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "11/01/1995"
    end_train_start_test_date = "11/07/1998"
    end_test_date = "11/01/2002"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types)
    for symbol in ["NDX"]:
        experiment(file_output_name="benchmarking_abraham", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def benchmark_abraham2004modeling_nsei():
    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(30, 10), num=15)).astype(int))
    n_seqs = list(np.ceil(np.logspace(math.log(1, 10), math.log(10, 10), num=1)).astype(int))

    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

    indicators = [["price"]]
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "01/01/1998"
    end_train_start_test_date = "15/12/1999"
    end_test_date = "01/12/2001"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types)
    for symbol in ["NSEI"]:
        experiment(file_output_name="benchmarking_abraham", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)


def benchmark_hansson2017stock():
    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(30, 10), num=15)).astype(int))
    n_seqs = [1]
    indicators = [["price"]]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "02/01/2009"
    end_train_start_test_date = "13/08/2014"
    end_test_date = "28/04/2017"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * 3
    for symbol in ["BVSP", "OMX"]:
        experiment(file_output_name="benchmarking_hansson", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

    ranked_ind = ["trix","mama","ad","ppo","trima","adx","minus_di","rsi","obv","natr","minus_dm","aroon","sar","cmo","stochrsi","stochf","wma","midprice","t3","macdext","rocr","ht_dcphase","roc","ht_phasor","ht_dcperiod","ht_sine","dema","aroonosc","sma","bop","apo","adosc","willr","mfi","ultsoc","macd","dx","kama","trange","adxr","bbands","midpoint","ht_trendline","tema","ht_trendmode","stoch","plus_di","cci","plus_dm","ema","mom","price"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=4)).astype(int)]
    for symbol in ["SPX"]:
        experiment(file_output_name="benchmarking_hansson", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

def benchmark_gupta2012stock():
    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(30, 10), num=15)).astype(int))
    n_seqs = [1]
    indicators = [["price"]]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "02/01/2009"
    end_train_start_test_date = "13/08/2014"
    end_test_date = "28/04/2017"
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * 4
    for symbol in ["TISC"]:
        experiment(file_output_name="benchmarking_gupta", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)

    ranked_ind = ["trix","mama","ad","ppo","trima","adx","minus_di","rsi","obv","natr","minus_dm","aroon","sar","cmo","stochrsi","stochf","wma","midprice","t3","macdext","rocr","ht_dcphase","roc","ht_phasor","ht_dcperiod","ht_sine","dema","aroonosc","sma","bop","apo","adosc","willr","mfi","ultsoc","macd","dx","kama","trange","adxr","bbands","midpoint","ht_trendline","tema","ht_trendmode","stoch","plus_di","cci","plus_dm","ema","mom","price"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=4)).astype(int)]
    for symbol in ["AAPL", "IBM", "DELL"]:
        experiment(file_output_name="benchmarking_gupta", symbol=symbol, start_train_date=start_train_date, end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)


def benchmark_lin2009short():
    n_lags = list(np.ceil(np.logspace(math.log(1, 10), math.log(30, 10), num=15)).astype(int))
    n_seqs = [1]
    n_batches = ["full_batch"]  # , "half_batch", "online"]
    # http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1
    model_types = ["bi", "conv"]  # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
    start_train_date = "02/01/2009"
    end_train_start_test_date = "13/08/2014"
    end_test_date = "28/04/2017"


    ranked_ind = ["trix", "mama", "ad", "ppo", "trima", "adx", "minus_di", "rsi", "obv", "natr", "minus_dm", "aroon",
                  "sar", "cmo", "stochrsi", "stochf", "wma", "midprice", "t3", "macdext", "rocr", "ht_dcphase", "roc",
                  "ht_phasor", "ht_dcperiod", "ht_sine", "dema", "aroonosc", "sma", "bop", "apo", "adosc", "willr",
                  "mfi", "ultsoc", "macd", "dx", "kama", "trange", "adxr", "bbands", "midpoint", "ht_trendline", "tema",
                  "ht_trendmode", "stoch", "plus_di", "cci", "plus_dm", "ema", "mom", "price"]
    indicators = [ranked_ind[:x] for x in np.ceil(np.logspace(math.log(1, 10), math.log(52, 10), num=4)).astype(int)]
    symbols = ["AHC", "AMD","BBT", "CIEN","GD","HRB","IR","JCP","NBR","NSC","PBI","PPL"
               ,"PSA","RHI","SFA","SRE","THC","UIS","USB", "FDO", "CPN", "KMG"]
    global n_experiment
    n_experiment = len(n_lags) * len(n_seqs) * len(n_batches) * len(indicators) * len(model_types) * len(symbols)
    # Updated:   Outdated: "GTW","LXK", "ACE",
    for symbol in symbols:
        experiment(file_output_name="benchmarking_lin", symbol=symbol, start_train_date=start_train_date,
                       end_train_start_test_date=end_train_start_test_date,
                       end_test_date=end_test_date, n_lags=n_lags,
                       n_seqs=n_seqs, n_batches=n_batches, indicators=indicators, model_types=model_types)
