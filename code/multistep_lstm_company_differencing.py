from multistep_lstm_company import MultiStepLSTMCompany
from time import time
import pandas as pd



# No preprocessing in this stage, see if it is good
class MultiStepLSTMCompanyWithDifferencing(MultiStepLSTMCompany):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons, tech_indicators=[]):
        MultiStepLSTMCompany.__init__(self, name, train_start_date_string, train_end_test_start_date_string,
                                      test_end_date_string, n_lag, n_seq, n_epochs, n_batch, n_neurons, tech_indicators)
