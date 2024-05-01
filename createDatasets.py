# Description: This file is used to create the datasets for the model. It reads the files from the dataset and creates the dataset for the model.
from preprocessData import lexer
import os
import random
from relationsMatrix import get_relations_matrix
import tensorflow as tf
import numpy as np
from embedding import create_embedding


def get_plag_samples(dataset,labels):
    plag_samples = []
    for i in range(len(labels)):
        if labels[i] == 1:
            plag_samples.append(dataset[i])
    
    return plag_samples


# read files with plagarism
def getPlagPairs():
    plag = open("fire14-source-code-training-dataset/SOCO14-java.qrel").readlines()

    plagpairs = [] # list of plagarised pairs

    #each line is a pair of plagarised files
    for line in plag:
        # split the line into two files
        file1, file2 = line.split(' ')
        # strip/newline
        file2 = file2.strip( '\n')

        plagpairs.append([file1, file2])

    #tokenize the plag pairs

    tokenizedPlagPairs = []
    for pair in plagpairs:
        file1 = open('./fire14-source-code-training-dataset/java/' + str(pair[0])).read()
        file2 = open('./fire14-source-code-training-dataset/java/' + str(pair[1])).read()
        fileContent1 = lexer(file1)
        fileContent2 = lexer(file2)
        tokenizedPlagPairs.append([[pair[0], fileContent1], [pair[1], fileContent2], 1])

    return tokenizedPlagPairs

#generate random pairs
def getRandomPairs(matrix):
    tokenizedFiles=[]
    #tokenize all the files
    for file in os.listdir('./fire14-source-code-training-dataset/java/'):
        java_code = open('./fire14-source-code-training-dataset/java/' + file).read()

        fileContent = lexer(java_code)
        filenum = int(file.split('.')[0])


        tokenizedFiles.append([filenum,fileContent])

    #create random  pair dataset with all the files

    # Shuffle the list of tokenized files
    random.shuffle(tokenizedFiles)

    start = 0
    end = len(tokenizedFiles) - 1

    # Create pairs of files
    randomPairs = []
    while start < end:
        randomPairs.append((tokenizedFiles[start], tokenizedFiles[end], matrix[start][end]))
        start += 1
        end -= 1

    return randomPairs

def getDataSet(tokenizedPlagPairs, randomPairs, randPercentage=0.5,totalPairs=129):
    dataset = []
    # Add random pairs to the dataset
    limit = int(len(randomPairs) * randPercentage)

    for i in range(limit):
        dataset.append(randomPairs[i])
    # Add just the remaining percentage of plagarised pairs to the dataset
    remainingPlagPairs = totalPairs - limit
    for pair in tokenizedPlagPairs[0:remainingPlagPairs]:
        dataset.append(pair)

    # Shuffle the dataset
    random.shuffle(dataset)

    return dataset

# Get plag percentage
def getPlagPercentage(dataset, datasetName):
    plagCount = 0
    for pair in dataset:
        if pair[2] == 1:
            plagCount += 1

    plagPercentage = plagCount / len(dataset)
    print(f"{datasetName} plagarism percentage: {plagPercentage}")

def create_batches(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    return dataset

def create_dataset():
    #create matrix with plagarised pairs
    filename = "fire14-source-code-training-dataset/SOCO14-java.qrel"
    file_directory = "fire14-source-code-training-dataset/java"
    matrix = get_relations_matrix(filename, file_directory)

    # Get plagarised pairs
    plagPairs = getPlagPairs()
    # Get random pairs
    randomPairs = getRandomPairs(matrix)
    # Get dataset
    dataset = getDataSet(plagPairs, randomPairs, randPercentage=.5, totalPairs=129)



    # Get plagarism percentage
    getPlagPercentage(randomPairs, "Random Pairs")
    getPlagPercentage(plagPairs, "Plagarised Pairs")
    getPlagPercentage(dataset, "Dataset")


    #divide the dataset into data and labels
    data = []
    labels = []
    for pair in dataset:
        vector1, vector2 = create_embedding(pair[0][1], pair[1][1])
        data.append([vector1, vector2])
        labels.append(pair[2])

    # Convert data to ragged tensor
    ragged_data = tf.ragged.constant(data)

    # Pad ragged tensor
    padded_data = ragged_data.to_tensor()

    # Convert to tensor
    tensor_tf = tf.convert_to_tensor(padded_data)

    labels = tf.convert_to_tensor(labels)


    print("Tensor:", tensor_tf.shape)

    return tensor_tf, labels