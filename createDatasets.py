# Description: This file is used to create the datasets for the model. It reads the files from the dataset and creates the dataset for the model.
from preprocessData import lexer
import os
import random


#read files with plagarism



#create matrix with plagarised pairs


tokenizedFiles=[]
#do the same for all the files in the dataset
for file in os.listdir('./fire14-source-code-training-dataset/java/'):
    java_code = open('./fire14-source-code-training-dataset/java/' + file).read()
    
    fileContent = ''
    tokenLines = lexer(java_code)
    for Line in tokenLines:
        if Line != []:
          fileContent += ' '.join(Line) + '\n'
    # if file != 'BruteForce.java':
    filenum = int(file.split('.')[0])


    tokenizedFiles.append([filenum,fileContent])

print(len(tokenizedFiles))
    
#create random  pair dataset with all the files

# Shuffle the list of tokenized files
random.shuffle(tokenizedFiles)

start = 0
end = len(tokenizedFiles) - 1

# Create pairs of files
pairs = []
while start < end:
    pairs.append((tokenizedFiles[start], tokenizedFiles[end]))
    start += 1
    end -= 1

print(len(pairs))

print(pairs[0][0][0], pairs[0][1][0])
