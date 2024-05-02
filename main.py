from createDatasets import create_dataset, create_batches
from model import trainModel
import numpy as np
from test import test_model
from tensorflow.keras import models

def main():
    data, labels= create_dataset()
    data_reshaped = np.squeeze(data, axis=1)

    div_train = int(len(data) * 0.8)
    div_val = int(len(data) * 0.2)
    X_train = data[:div_train]
    X_val = data[div_train:div_train+div_val]
    y_train = labels[:div_train]
    y_val = labels[div_train:div_train+div_val]

    train_data = create_batches(X_train, y_train)
    validation_data = create_batches(X_val, y_val)

    trainModel(train_data, validation_data)
    #model = models.load_model('RNN.keras')
    #test_model(model, X_test, y_test)

main()
