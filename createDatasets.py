# Description: This file is used to create the datasets for the model. It reads the files from the dataset and creates the dataset for the model.
from preprocessData import lexer
import os
import random
from relationsMatrix import get_relations_matrix
import tensorflow as tf
import numpy as np
from embedding import create_embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


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
    text1=[]
    text2=[]
    for pair in dataset:
        # vector1, vector2 = create_embedding(pair[0][1], pair[1][1])
        # data.append([vector1 + vector2])
        text1.append(pair[0][1])
        text2.append(pair[1][1])
        labels.append(pair[2])

    data = text1 + text2
    max_seq_length = max(len(seq) for seq in data)

    print("Max sequence length:", max_seq_length)

    text1_padded = pad_sequences(text1, maxlen=max_seq_length, padding='post')
    text2_padded = pad_sequences(text2, maxlen=max_seq_length, padding='post')

    labels = tf.convert_to_tensor(labels)

    text1_indices = np.array(text1_padded)
    text2_indices = np.array(text2_padded)


        # Split the data into training, validation, and test sets
    pair1_train, pair1_test, pair2_train, pair2_test, labels_train, labels_test = train_test_split(
        text1_indices, text2_indices, labels, test_size=0.2, random_state=42)

    pair1_train, pair1_val, pair2_train, pair2_val, labels_train, labels_val = train_test_split(
        text1_indices, text2_indices, labels_train, test_size=0.2, random_state=42)
    
    pair1_train = np.array(pair1_train)
    pair2_train = np.array(pair2_train)
    pair1_val = np.array(pair1_val)
    pair2_val = np.array(pair2_val)
    pair1_test = np.array(pair1_test)
    pair2_test = np.array(pair2_test)

    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)
    labels_test = np.array(labels_test)
    

    return (pair1_train, pair2_train, labels_train), (pair1_val, pair2_val, labels_val), (pair1_test, pair2_test, labels_test)