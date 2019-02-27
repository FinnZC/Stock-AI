




def get_optimal_epochs_batch_neurons_params(symbol, start_train_date, end_train_start_test_date, end_test_date, n_lag, n_seq, n_epochs, n_batch, n_neurons):
    # This is optimising parameters for n_epochs, n_batch, and n_neurons
    # param = {"n_epochs": n_epochs, "n_batch": n_batch, "n_neurons": n_neurons}

    for n_e in n_epochs:
        for n_b in n_batch:
            for n_n in n_neurons:
                model = MultiStepLSTMCompany(symbol, start_train_date, end_train_start_test_date, end_test_date, n_lag, n_seq, n_e, n_b, n_n)
                model.train()
                model.


