from createDatasets import create_dataset, create_batches
from model import trainModel
import numpy as np
from test import test_model
from keras.models import load_model

def main():
    train,val,test= create_dataset()


   

    trainModel(train,val)

    model = load_model('RNN.keras')
    test_model(model, X_test, y_test)

main()
