import os
filename = "fire14-source-code-training-dataset/SOCO14-java.qrel"
file_directory = "fire14-source-code-training-dataset/java"

def get_relations_matrix(filename, file_directory):
    # Get the list of files in the directory
    file_list = os.listdir(file_directory)
    matrix_size = len(file_list)
    matrix = [[0] * matrix_size for _ in range(matrix_size)]

        # Define the file names and their relations
    with open(filename, "r") as file:
        for line in file:
            file_names = line.split(' ')
            i = int(file_names[0].split('.')[0])
            j = int(file_names[1].split('.')[0])
            matrix[i][j] = 1
            matrix[j][i] = 1

    return matrix
