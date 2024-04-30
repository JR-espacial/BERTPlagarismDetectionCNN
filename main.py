from createDatasets import createDataset, create_batches
from Model import trainModel
import numpy as np

def main():
    data, labels= createDataset()
    data_reshaped = np.squeeze(data, axis=1)
    
    div_train = int(len(data_reshaped) * 0.6)
    div_val = int(len(data_reshaped) * 0.2)
    X_train = data_reshaped[:div_train]
    X_val = data_reshaped[div_train:div_train+div_val]
    X_test = data_reshaped[div_val:]
    y_train = labels[:div_train]
    y_val = labels[div_train:div_train+div_val]
    y_test = labels[div_val:]
    
    train_data = create_batches(X_train, y_train)
    validation_data = create_batches(X_val, y_val)
    
    trainModel(train_data, validation_data)


main()
