import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

def trainModel(data,labels,val_data, val_labels):

  vocab_size = getNumTokens()

  # Reshape the data remove the second dimension
  data_reshaped = np.squeeze(data, axis=1)
  val_reshaped = np.squeeze(val_data, axis=1)


  model = createModel( VOCAB_SIZE = vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(data_reshaped, labels, epochs=10, batch_size=10, validation_data=(val_reshaped, val_labels))

  #save model
  model.save('RNN.keras')

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