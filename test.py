def test_model(model, x_test, y_test):
  # Evaluar el modelo en el conjunto de prueba
  test_loss, test_acc = model.evaluate(x_test, y_test)
  # Print the test loss and accuracy
  print("Test Accuracy:", test_acc)
  print("Test Loss:", test_loss)