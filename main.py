from createDatasets import create_dataset, create_batches
from Model import trainModel
import numpy as np
from test import test_model
from keras.models import load_model

def execute(function_name, function, *args):
    #prompt the use if he wants to ecexute the function name
    opt = input("Do you want to " + function_name + " ? (y/n)")
    if opt == "y":
        output = function(*args)
        if output != None:
            return output
    else:
        print("Function " + function_name + " was not executed")


def main():
    data, labels= create_dataset()
    # data_reshaped = np.squeeze(data, axis=1)

    div_train = int(len(data) * 0.7)
    div_val = int(len(data) * 0.1)
    # test would be the 20% remaining

    X_train = data[:div_train]
    X_val = data[div_train:div_train+div_val]
    X_test = data[div_train+div_val:]

    y_train = labels[:div_train]
    y_val = labels[div_train:div_train+div_val]
    y_test = labels[div_train+div_val:]

    # train_data = create_batches(X_train, y_train)
    # validation_data = create_batches(X_val, y_val)


    execute("train the model", trainModel, X_train, y_train, X_val, y_val)

    model = load_model('RNN.keras')

    execute("test the model",test_model, model, X_test, y_test)

main()
