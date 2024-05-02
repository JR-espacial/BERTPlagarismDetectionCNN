import tensorflow as tf
from preprocessData import getNumTokens
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def createModel(VOCAB_SIZE, EMBEDDING_DIM=70, MAX_SEQUENCE_LENGTH=27):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,70,mask_zero=True),
        tf.keras.layers.GRU(units =100),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def trainModel(data, labels, val_data, val_labels):
  vocab_size = getNumTokens()

  # Reshape the data remove the second dimension
  data_reshaped = np.squeeze(data, axis=1)
  val_reshaped = np.squeeze(val_data, axis=1)


  model = createModel( VOCAB_SIZE = vocab_size)
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  history = model.fit(
      data_reshaped,
      labels,
      epochs=30,
      batch_size=10,
      validation_data=(val_reshaped, val_labels)
  )

  #save model
  model.save('RNN.keras')

  plotModel(history)

  train_predictions = (model.predict(data_reshaped) > 0.5).astype("int32")
  plot_confusion_matrix(labels, train_predictions, title='Training Confusion matrix')



# Plot model accuracy , loss
def plotModel(history):
    # Plot model accuracy
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot model loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()


def plot_confusion_matrix(labels, predictions, title='Confusion matrix'):
    cm = confusion_matrix(labels, predictions)
    TN, FP, FN, TP = cm.ravel()

    # Calculo de metricas
    true_positive_rate = TP / (TP + FN)
    false_positive_rate = FP / (FP + TN)
    accuracy = TP+TN / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print("\n--------- Métricas ---------")
    print("Accuracy:", accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('True positive rate:', true_positive_rate)
    print('False positive rate:', false_positive_rate)

    # Graficar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()
