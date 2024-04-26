from createDatasets import createDataset
from Model import trainModel

def main():
    data, labels, mask = createDataset()
    
    trainModel(data, labels, mask)


main()
