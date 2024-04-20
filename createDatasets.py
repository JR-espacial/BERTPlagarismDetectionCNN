# Description: This file is used to create the datasets for the model. It reads the files from the dataset and creates the dataset for the model.
from preprocessData import lexer
import os
import random
from relationsMatrix import get_relations_matrix
import tensorflow as tf


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
def getRadomPairs(matrix):
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
def getPlagPercentage(dataset,datasetName):
    plagCount = 0
    for pair in dataset:
        if pair[2] == 1:
            plagCount += 1

    plagPercentage = plagCount / len(dataset)
    print(f"{datasetName} plagarism percentage: {plagPercentage}")


#create matrix with plagarised pairs
filename = "fire14-source-code-training-dataset/SOCO14-java.qrel"
file_directory = "fire14-source-code-training-dataset/java"
matrix = get_relations_matrix(filename, file_directory)

# Get plagarised pairs
plagPairs = getPlagPairs()
# Get random pairs
randomPairs = getRadomPairs(matrix)
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
    data.append([pair[0][1], pair[1][1]])
    labels.append(pair[2])

print("Data:", data[0])
print("Labels:", labels[0])

tensor_tf = tf.convert_to_tensor(data, dtype=tf.string)

print("Tensor:", tensor_tf.shape)