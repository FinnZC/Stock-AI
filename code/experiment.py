
import numpy as np
n_epochs = np.logspace(100, 10000, num=10)
n_batches = np.logspace(1, 100, num=10)
n_neurons = np.logspace(1, 100, num=10)


def get_optimal_epochs_batch_neurons_params(symbol, start_train_date, end_train_start_test_date, end_test_date, n_lag, n_seq, n_epochs, n_batches, n_neurons):
    # This is optimising parameters for n_epochs, n_batch, and n_neurons
    # param = {"n_epochs": n_epochs, "n_batch": n_batch, "n_neurons": n_neurons}
    accuracy = dict()
    for n_e in n_epochs:
        accuracy[n_e] = dict()
        for n_b in n_batches:
            accuracy[n_b] = dict()
            for n_n in n_neurons:
                accuracy[n_n] = dict()
                obj = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date, end_test_date, n_lag, n_seq, n_e, n_b, n_n)
                obj.train()
                predictions = obj.predict()
                trend_score = obj.score(metric="trend", predictions=predictions)
                lstm_score = obj.score(metric="rmse", predictions=predictions)
                accuracy[n_e][n_b][n_n] = (trend_score, lstm_score)



