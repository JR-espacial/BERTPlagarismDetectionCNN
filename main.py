import preprocessData
import Model
import train
import test

def main():
    data = preprocessData.load_data()
    model = Model.Bert(data)
    train.trainModel(model, data)
    test.testModel(model, data)

