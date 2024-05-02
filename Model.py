import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Bidirectional, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np

# Define the Siamese recurrent network architecture
def siamese_rnn(input_shape, lstm_units=64, embedding_dim=128):
  # Define input layer
  input_code = Input(shape=input_shape)

  # Define LSTM layer
  lstm_layer = LSTM(lstm_units)(input_code)

  # Define output layer
  output = Dense(embedding_dim, activation='relu')(lstm_layer)

  # Build the model
  model = Model(inputs=input_code, outputs=output)

  return model

# Define function to compute similarity/distance between embeddings
def compute_distance(embeddings):
  x1, x2 = embeddings
  distance = tf.reduce_sum(tf.abs(x1 - x2), axis=1, keepdims=True)
  return distance

# Define function to create Siamese network
def create_siamese_model(input_shape):
  # Define input layers for two code snippets
  input_code1 = Input(shape=input_shape)
  input_code2 = Input(shape=input_shape)

  # Create two identical Siamese recurrent network branches
  siamese_branch = siamese_rnn(input_shape)

  # Encode both code snippets using the Siamese branches
  encoded_code1 = siamese_branch(input_code1)
  encoded_code2 = siamese_branch(input_code2)

  # Compute distance between the encoded embeddings
  distance = Lambda(compute_distance)([encoded_code1, encoded_code2])

  # Define output layer
  output = Dense(1, activation='sigmoid')(distance)

  # Build the Siamese model
  siamese_model = Model(inputs=[input_code1, input_code2], outputs=output)

  return siamese_model


def createModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=27):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(VOCAB_SIZE, 50, mask_zero=True),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(VOCAB_SIZE,70,mask_zero=True),
      tf.keras.layers.GRU(units =100),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def trainModel(data, labels, val_data, val_labels):
  VOCAB_SIZE = 42
  MAX_SEQUENCE_LENGTH = 2381

  data_1, data_2 = data
  val_data_1, val_data_2 = val_data

  print("Shape:", data_1.shape)
  # Define input shape (sequence length, vocabulary size)
  input_shape = (MAX_SEQUENCE_LENGTH, VOCAB_SIZE)  # Example: sequences of length 100, vocabulary size of 500
  siamese_model = create_siamese_model(input_shape)
  siamese_model.summary()

  model = create_siamese_model(input_shape)

  # Compile the model
  siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Train the model
  history = siamese_model.fit([data_1, data_2], labels,
                              validation_data=([val_data_1, val_data_2], val_labels),
                              epochs=10, batch_size=10, verbose=1)



  # Compile and train the model
  # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  # model.fit([data_1, data_2], labels, epochs=10, batch_size=10, validation_data=([val_data_1, val_data_2], val_labels))

  #save model
  model.save('Siamese.keras')

  plotModel(model)
  # plot_confusion_matrix(val_labels, model.predict(val_reshaped))



# Plot model accuracy , loss
def plotModel(model):
    import matplotlib.pyplot as plt

    # Plot model accuracy
    plt.plot(model.history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot model loss
    plt.plot(model.history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
