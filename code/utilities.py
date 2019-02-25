import pickle
from keras.models import load_model
import holidays


def load_lstm_model(file_name):
    model = load_model("models/" + file_name + ".h5")
    return model


def load_object(file_name):
    with open('obj/' + file_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_saved_model(file_name):
    obj = load_object(file_name)
    obj.lstm_model = load_lstm_model(file_name)
    obj.us_holidays = holidays.UnitedStates()

    return obj