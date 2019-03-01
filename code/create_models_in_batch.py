import os
import pandas as pd
from multistep_lstm_company import MultiStepLSTMCompany

nasdaq_100_dict = dict()
start_train_date = "01/01/2017"
end_train_start_test_date = "10/01/2017"
end_test_date = "20/01/2017"

def get_all_data():

    data = pd.read_csv(os.path.join(os.getcwd(), 'symbols', 'nasdaq100list_feb2019.csv'))
    nasdaq_100_symbols = data["Symbol"].values.tolist()
    for symbol in nasdaq_100_symbols:
        print("STARTING DOWNLOADING FOR", symbol)
        nasdaq_100_dict[symbol] = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date, end_test_date,
                                n_lag=1, n_seq=1, n_epochs=3000, n_batch=1, n_neurons=4, tech_indicators="all")
        nasdaq_100_dict[symbol].save()
        nasdaq_100_dict[symbol].raw_pd.to_csv("raw_data/" + symbol + "_raw_pd.csv")


