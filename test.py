from model import plot_confusion_matrix
from tensorflow.keras import models

def test_model(model, x_test, y_test):
  # Evaluar el modelo en el conjunto de prueba
  test_loss, test_acc = model.evaluate(x_test, y_test)
  # Print the test loss and accuracy
  print("Test Accuracy:", test_acc)
  print("Test Loss:", test_loss)

  test_predictions = (model.predict(x_test) > 0.5).astype("int32")
  plot_confusion_matrix(y_test, test_predictions)
