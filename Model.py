import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from preprocessData import getNumTokens
import numpy as np

def createSiameseModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=27):
    model = tf.keras.Sequential([
        LSTM(128),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()
    return model

def trainModel(train_data, validation_data):
    vocab_size = getNumTokens()
    sequence_length = train_data.element_spec[0].shape[1]
    model = createSiameseModel(VOCAB_SIZE=vocab_size, MAX_SEQUENCE_LENGTH=sequence_length)

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=train_data,  # Using the same data for both branches of the Siamese network
              epochs=30,
              batch_size=32,
              validation_data=validation_data)

    plotModel(model)

    # Save model
    model.save('SiameseGRU.keras')


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
