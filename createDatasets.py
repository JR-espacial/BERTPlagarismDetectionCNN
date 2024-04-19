# Description: This file is used to create the datasets for the model. It reads the files from the dataset and creates the dataset for the model.
from preprocessData import lexer

#read file with plagarism



#create matrix with plagarised pairs


#create random datsaet with all the files




# Example usage: load file
java_code = open('./fire14-source-code-training-dataset/java/043.java').read()
outf = open('output2.txt', 'a')

tokenLines = lexer(java_code)
for Line in tokenLines:
    print("Line:", Line)
    # Write to file
    outf.write(' '.join(Line) + '\n')
