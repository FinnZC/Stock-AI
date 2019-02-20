from multistep_lstm_company import MultiStepLSTMCompany
from alpha_vantage.techindicators import TechIndicators
from time import time

class MultiStepLSTMCompanyTechIndicators(MultiStepLSTMCompany):
    def __init__(self, name, train_start_date_string, train_end_test_start_date_string, test_end_date_string,
                 n_lag, n_seq, n_epochs, n_batch, n_neurons):
        MultiStepLSTMCompany.__init__(self, name, train_start_date_string, train_end_test_start_date_string,
                                      test_end_date_string,
                                      n_lag, n_seq, n_epochs, n_batch, n_neurons)

        self.tech_indicators = TechIndicators(key='3OMS720IM6CRC3SV', output_format='pandas', indexing_type='date')

