import re
from numpy.core.multiarray import array
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocessData import lexer
from collections import defaultdict

def text_words(texto1, texto2):
  text1_words = re.sub(r'[^\w\s\d]', '', texto1).split()
  text2_words = re.sub(r'[^\w\s\d]', '', texto2).split()

  return text1_words, text2_words


# Genera la matriz de transición de un texto dado
def generate_transition_matrix(text, unique, label):
  words_set = list(unique.keys())
  N = len(unique)

  # Se reduce en 1 la cantidad de la ultima palabra del texto porque el último elemento no tiene transición
  unique[text[-1]][label] -= 1


  # Creación de la matriz en 0
  transition_matrix = np.zeros((N, N))

  for i in range(len(text)-1):
    current_word = text[i]
    current_word_index = words_set.index(current_word)
    next_word = text[i + 1]
    next_word_index = words_set.index(next_word)

    # Incrementa la transición en las celdas que corresponden
    transition_matrix[current_word_index][next_word_index] += 1


  # Saca la probabilidad con la que aparece cada palabra
  for i in range(len(words_set)):
    for j in range(len(words_set)):
      if unique[words_set[i]][label] > 0:
        transition_matrix[i][j] /= unique[words_set[i]][label]

  return transition_matrix

# Genera vectores de frecuencias para ambos textos
def frequency_vectors(unique):
  vector_original = []
  vector_sintetico = []

  for key in unique:
    vector_original.append(unique[key][0])
    vector_sintetico.append(unique[key][1])

  return vector_original, vector_sintetico

# Genera un diccionario con las palabras y sus frecuencias en cada texto
def word_dictionary(text1_words, text2_words):
  unique = {}
  for word in text1_words:
    if unique.get(word) is None:
      unique[word] = [0, 0]
      unique[word][0] += 1
    else:
      unique[word][0] += 1
  for word in text2_words:
    if unique.get(word) is None:
      unique[word] = [0, 0]
      unique[word][1] += 1
    else:
      unique[word][1] += 1
  return unique


# -------- Representación vectorial con frecuencias --------

def frequency_similarity(unique):
  # Generar vectores de frecuencia para ambos textos
  vector_original, vector_sintetico = frequency_vectors(unique)

  for key in unique:
    vector_original.append(unique[key][0])
    vector_sintetico.append(unique[key][1])

  # Calcular la similitud con el coseno entre los dos vectores
  vector1 = np.array(vector_original).reshape(1, -1)
  vector2 = np.array(vector_sintetico).reshape(1, -1)

  similitudFrecuencias = cosine_similarity(vector1, vector2)

  return similitudFrecuencias[0][0]
  # return (vector_original, vector_sintetico)


# -------- Representación vectorial con TF-IDF --------

def tfidf_similarity(unique, text1_words, text2_words):
  vector_original, vector_sintetico = frequency_vectors(unique)

  # Sacar los TF's de cada texto
  tf1, tf2 = [], []
  for num in vector_original:
    tf1.append(num / len(text1_words))
  for num in vector_sintetico:
    tf2.append(num / len(text2_words))

  # Sacar IDF's de cada palabra
  idf = []

  def num_docs_word(word, unique):
    if unique.get(word) is None:
      return 0
    elif unique[word][0] > 0 and unique[word][1] > 0:
      return 2
    else:
      return 1

  for word in unique:
    idf.append(np.log(2 / (num_docs_word(word, unique) + 1)) + 1)

  # Calcular TF-IDF de cada palabra
  tfidf1 = []
  tfidf2 = []

  for i in range(len(tf1)):
    tfidf1.append(tf1[i] * idf[i])
    tfidf2.append(tf2[i] * idf[i])

  # Convertir los vectores en matrices de una fila
  tfidf1 = np.array(tfidf1).reshape(1, -1)
  tfidf2 = np.array(tfidf2).reshape(1, -1)

  # Calcular el coseno entre los dos vectores
  similitud = cosine_similarity(tfidf1, tfidf2)
  return similitud[0][0]
  # return (tfidf1, tfidf2)

# -------- Coseno entre 2 matrices --------
def cosine_angle_between_matrixes(A, B):
  # Transpuesta de B
  BT = np.transpose(B)

  # Matriz C
  C = np.dot(BT, A)

  # Producto interno
  prod_int = np.trace(C)

  # Normalizacion de matriz A
  normA = np.sqrt(np.trace(np.transpose(A) @ A))

  # Normalizacion de matriz B
  normB = np.sqrt(np.trace(np.transpose(B) @ B))

  # Coseno del angulo entre A y B
  cos_ang = prod_int / (normA * normB)

  return cos_ang


def create_embedding(text1, text2, option = "2"):
  unique = word_dictionary(text1, text2)

  similitud = 0

  if option == "1":
    vec1, vec2 = frequency_similarity(unique)
    # similitud = cosine_similarity(vec1, vec2)[0][0]

  elif option == "2":
    vec1, vec2 = tfidf_similarity(unique, text1, text2)
    # similitud = cosine_similarity(vec1, vec2)[0][0]

  elif option == "3":
    vec1 = generate_transition_matrix(text1, unique,0)
    vec2 = generate_transition_matrix(text2, unique,1)
    # similitud = cosine_angle_between_matrixes(vec1, vec2)

  print(vec1, vec2)
  return (vec1, vec2)


def getSimilarities(data_pairs):
  print("Obteniendo similitudes...")
  allSimilarities = []
  for pair in data_pairs:
    similarities = []
    text1, text2 = pair[0], pair[1]
    unique = word_dictionary(text1, text2)

    transition_matrix_1 = generate_transition_matrix(text1, unique, 0)
    transition_matrix_2 = generate_transition_matrix(text2, unique, 1)
    similarities.append(cosine_angle_between_matrixes(transition_matrix_1, transition_matrix_2))

    allSimilarities.append(similarities)

  print("Similitudes del primero: ", allSimilarities[0])
  print("Similitudes del segundo: ", allSimilarities[1])
  return allSimilarities
