from Model import plot_confusion_matrix
import tensorflow as tf
import numpy as np


def test_model(model, x_test, y_test):
  x_test = np.squeeze(x_test, axis=1)

  # Evaluar el modelo en el conjunto de prueba
  test_loss, test_acc = model.evaluate(x_test, y_test)

  # Print the test loss and accuracy
  print("Test Accuracy:", test_acc)
  print("Test Loss:", test_loss)

  test_predictions = (model.predict(x_test) > 0.5).astype("int32")

  plot_confusion_matrix(y_test, test_predictions, title='Test Confusion Matrix')