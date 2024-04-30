import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np

def createModel(VOCAB_SIZE,EMBEDDING_DIM=70, HIDDEN_UNITS=3, NUM_CLASSES=2):
  # Define your RNN model with masking
#   model = tf.keras.Sequential([
#         tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),
#         tf.keras.layers.SimpleRNN(units=HIDDEN_UNITS),
#         tf.keras.layers.Flatten(),  # Flatten the output
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
#     ])
  

  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,70,mask_zero=True),
    tf.keras.layers.GRU(units =100),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
  return model


def trainModel(X_train, X_test, y_train, y_test, mask):

  vocab_size = getNumTokens()

  model = createModel( VOCAB_SIZE=vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X_train, 
            y_train, 
            epochs=30, 
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
