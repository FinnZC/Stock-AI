import pickle
from keras.models import load_model
import holidays
import tensorflow as tf
import os
from keras import backend as K
import multiprocessing

NUM_PARALLEL_EXEC_UNITS = multiprocessing.cpu_count()


def optimal_tensorflow_config():
    # source https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus

    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

    session = tf.Session(config=config)
    K.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"



def load_lstm_model(file_name):
    model = load_model("models/" + file_name + ".h5")
    return model


def load_object(file_name):
    with open('objects/' + file_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_saved_model(file_name):
    obj = load_object(file_name)
    obj.lstm_model = load_lstm_model(file_name)
    obj.us_holidays = holidays.UnitedStates()
    return obj

#optimal_config()