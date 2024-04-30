import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np

def createModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=4630):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, mask_zero=True),
        tf.keras.layers.GRU(units=128, dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def trainModel(X_train, X_test, y_train, y_test):

  vocab_size = getNumTokens()

  model = createModel( VOCAB_SIZE=vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X_train, 
            y_train, 
            epochs=20, 
            batch_size=10,
            validation_data=(X_test, y_test))

  plotModel(model)

  #save model
  model.save('RNN.keras')


#plot model accuracy , loss
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
