import os

def get_relations_matrix(filename, file_directory):
    # Define the file names and their relations
    with open(filename, "r") as file:
        file_relations = [line.strip().split() for line in file]

        # Get the list of files in the directory
        file_list = os.listdir(file_directory)

        # Sort the file list
        file_list.sort()

        # Create an empty matrix
        matrix_size = len(file_list)
        matrix = [[0] * matrix_size for _ in range(matrix_size)]

        # Populate the matrix
        for i in range(matrix_size):
            for j in range(matrix_size):
                # Check if the files are related
                if any(file_list[i] in relation for relation in file_relations):
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        
        return matrix
    
# "fire14-source-code-training-dataset/java"
# fire14-source-code-training-dataset/SOCO14-java.qrel