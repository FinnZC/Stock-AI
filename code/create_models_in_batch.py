from multistep_lstm_company import MultiStepLSTMCompany
nasdaq_100_symbols = list()
nasdaq_100_dict = dict()
start_train_date = "01/01/2017"
end_train_start_test_date = "10/01/2017"
end_test_date = "20/01/2017"

def get_all_data():
    for symbol in nasdaq_100_symbols:
        nasdaq_100_dict[symbol] = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date, end_test_date,
                                n_lag=1, n_seq=1, n_epochs=3000, n_batch=1, n_neurons=4, tech_indicators="all")
        nasdaq_100_dict[symbol].save()


