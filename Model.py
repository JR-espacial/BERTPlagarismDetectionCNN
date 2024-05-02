import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Embedding
from preprocessData import getNumTokens
import numpy as np

def createModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=5794):
    model = tf.keras.Sequential([
        GRU(128, return_sequences=True),
        GRU(64, return_sequences=True),
        GRU(32),
        Dense(1, activation='sigmoid')
    ])
    model.summary()
    return model

def trainModel(train_data, validation_data):

  vocab_size = getNumTokens()

  model = createModel( VOCAB_SIZE = vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(train_data, epochs=20, batch_size=32, validation_data=validation_data)

  plotModel(model)

  #save model
  model.save('RNN.h5')


# Plot model accuracy , loss
def plotModel(model):
	import matplotlib.pyplot as plt

	# Plot model accuracy
	plt.plot(model.history.history['accuracy'], label='Train')
	plt.plot(model.history.history['val_accuracy'], label='Validation')
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train'], loc='upper left')
	plt.show()

	# Plot model loss
	plt.plot(model.history.history['loss'], label='Train')
	plt.plot(model.history.history['val_loss'], label='Validation')
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train'], loc='upper left')
	plt.show()
