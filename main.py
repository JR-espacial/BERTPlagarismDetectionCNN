from createDatasets import create_dataset, create_batches
from model import trainModel
import numpy as np
from test import test_model
from keras.models import load_model

def main():
    data_1, data_2, labels = create_dataset()
    # data_reshaped = np.squeeze(data, axis=1)

    div_train = int(len(labels) * 0.8)
    div_val = int(len(labels) * 0.2)

    train_data_1 = data_1[:div_train]
    train_data_2 = data_2[:div_train]
    train_labels = labels[:div_train]

    val_data_1 = data_1[div_train : div_train + div_val]
    val_data_2 = data_2[div_train : div_train + div_val]
    val_labels = labels[div_train : div_train + div_val]

    # train_data = create_batches(X_train, y_train)
    # validation_data = create_batches(X_val, y_val)

    trainModel((train_data_1, train_data_2), train_labels, (val_data_1, val_data_2), val_labels)

    model = load_model('RNN.keras')
    test_model(model, (val_data_1, val_data_2), val_labels)

main()
