import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np

def createSiameseModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=27):
    input_layer = tf.keras.layers.Input(shape=(2, MAX_SEQUENCE_LENGTH,))
    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
    reshape = tf.keras.layers.Reshape((2, MAX_SEQUENCE_LENGTH, 70))

    # Shared GRU layer
    shared_gru_layer = tf.keras.layers.GRU(units=128, dropout=0.1)

    # Branch 1
    branch_1 = embedding_layer(input_layer)
    branch_1 = reshape(branch_1)
    branch_1 = shared_gru_layer(branch_1)

    # Branch 2
    branch_2 = embedding_layer(input_layer)
    branch_2 = reshape(branch_2)
    branch_2 = shared_gru_layer(branch_2)

    # Calculate L1 distance
    distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([branch_1, branch_2])

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)

    model = tf.keras.models.Model(inputs=[input_layer, input_layer], outputs=output)
    model.summary()
    return model

def trainModel(train_data, validation_data):
    vocab_size = getNumTokens()
    model = createSiameseModel(VOCAB_SIZE=vocab_size)
    train_data, train_labels = train_data
    validation_data, validation_labels = validation_data

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=[train_data, train_data],  # Using the same data for both branches of the Siamese network
              y=train_labels,  # Assuming train_labels contains the labels for plagiarism
              epochs=20,
              batch_size=32,
              validation_data=([validation_data, validation_data], validation_labels))  # Assuming validation_labels contains the labels for validation data

    plotModel(model)

    # Save model
    model.save('SiameseGRU.keras')


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
