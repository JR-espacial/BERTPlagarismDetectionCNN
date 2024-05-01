import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np

def createModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=27):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 100, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def trainModel(data,labels,val_data, val_labels):

  vocab_size = getNumTokens()

  # Reshape the data remove the second dimension
  data_reshaped = np.squeeze(data, axis=1)
  val_reshaped = np.squeeze(val_data, axis=1)
  

  model = createModel( VOCAB_SIZE = vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(data_reshaped, labels, epochs=30, batch_size=10, validation_data=(val_reshaped, val_labels))

  plotModel(model)

  #save model
  model.save('RNN.h5')


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
