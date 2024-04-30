from createDatasets import createDataset
from Model import trainModel
import numpy as np

def main():
    data, labels, mask = createDataset()
    data_reshaped = np.squeeze(data, axis=1)
    
    div = int(len(data_reshaped) * 0.8)
    X_train = data_reshaped[:div]
    X_test = data_reshaped[div:]
    y_train = labels[:div]
    y_test = labels[div:]
    
    trainModel(X_train, X_test, y_train, y_test, mask)


main()
