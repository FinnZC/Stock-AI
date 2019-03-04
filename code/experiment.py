import numpy as np
import csv
import os
from multistep_lstm_company import MultiStepLSTMCompany
from datetime import date
import math


def get_optimal_epochs_batch_neurons_params(symbol, start_train_date, end_train_start_test_date, end_test_date, n_lags, n_seqs,
                                            n_epochs, n_batches, n_neurons, indicators, model_types):
    # This is optimising parameters for n_epochs, n_batch, and n_neurons
    # param = {"n_epochs": n_epochs, "n_batch": n_batch, "n_neurons": n_neurons}
    csv_columns = ["Company", "LSTM Type", "n_epoch", "n_neuros", "n_batch",
                   "n_lag", "n_seq", "Training Time",
                   "Indicator Number", "Indicators", "Trained Date",
                   "Start Train Date", "End Train/Start Test Date", "End Test Date",
                   "Model Name"]
    for i in range(30):
        csv_columns.append("Trend_t+" + str(i+1))
        csv_columns.append("APRE_t+" + str(i+1))
        csv_columns.append("RMSE_t+" + str(i+1))

    if not os.path.isdir("./experiments"):
        # create directory
        os.mkdir("experiments")

    filename = "./experiments/optimisation.csv"
    if not os.path.isfile(filename):
        # create new file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    for n_e in n_epochs:
        for n_b in n_batches:
            for n_n in n_neurons:
                for n_l in n_lags:
                    for n_s in n_seqs:
                        for m_t in model_types:
                            print("In process of training", symbol, "model type:", m_t, "n_lag:",
                                  n_l, "n_seq:", n_s, "n_epoch:", n_e, "n_batch:", n_b, "n_neurons:", n_n)
                            obj = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date,
                                                       end_test_date, n_lag=n_l, n_seq=n_s, n_epochs=n_e,
                                                       n_neurons=n_n, n_batch=n_b, tech_indicators=indicators,
                                                       model_type=m_t)
                            obj.train()
                            predictions = obj.predict()
                            trend_score = obj.score(metric="trend", predictions=predictions)
                            lstm_score = obj.score(metric="rmse", predictions=predictions)
                            apre_score = obj.score(metric="apre", predictions=predictions)

                            dic = {"Company": symbol,
                                   "LSTM Type": obj.model_type,
                                   "n_epoch": obj.n_epochs,
                                   "n_neuros": obj.n_neurons,
                                   "n_batch": obj.n_batch,
                                   "n_lag": obj.n_lag,
                                   "n_seq": obj.n_seq,
                                   "Training Time": obj.time_taken_to_train,
                                   "Start Train Date": obj.train_start_date_string,
                                   "End Train/Start Test Date": obj.train_end_test_start_date_string,
                                   "End Test Date": obj.test_end_date_string,
                                   "Indicator Number": len(obj.input_tech_indicators_list) + 1,
                                   "Indicators": "Share Price," + ",".join(obj.input_tech_indicators_list),
                                   "Trained Date": str(date.today()),
                                    "Model Name": obj.create_file_name()}
                            for i in range(n_s):
                                dic["Trend_t+" + str(i+1)] = trend_score[i]
                                dic["APRE_t+" + str(i+1)] = apre_score[i]
                                dic["RMSE_t+" + str(i+1)] = lstm_score[i]

                            append_dict_to_csv(filename, csv_columns, dic)

def append_dict_to_csv(csv_file_name, csv_columns, dic):
    with open(csv_file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(dic)


n_lags = [5]
n_seqs = [1, 3]
n_epochs = np.logspace(math.log(100, 10), math.log(5000, 10), num=10).astype(int)
n_neurons = np.logspace(math.log(1, 10), math.log(52, 10), num=10).astype(int)
n_batches = ["full_batch"]#, "half_batch", "online"]
#http://firsttimeprogrammer.blogspot.com/2015/09/selecting-number-of-neurons-in-hidden.html?m=1

indicators = "all"
model_types = ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] # ["vanilla", "stacked", "stacked", "bi", "cnn", "conv"] #
start_train_date = "01/12/2017"
end_train_start_test_date = "01/01/2018"
end_test_date = "01/04/2019"
pred = get_optimal_epochs_batch_neurons_params("AMZN", start_train_date, end_train_start_test_date, end_test_date, n_lags, n_seqs, n_epochs, n_batches, n_neurons, indicators, model_types)