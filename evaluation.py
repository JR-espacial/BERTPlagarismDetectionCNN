from Model import plot_confusion_matrix
import numpy as np

def complete_evaluation(model, data, labels, similarities):
  data = np.squeeze(data, axis=1)
  prediction_confidence = model.predict(data)
  results = []
  for i in range(len(prediction_confidence)):
    similarity_average = sum(similarities[i]) / len(similarities[i])
    model_average = (similarity_average + prediction_confidence[i]) / 2
    prediction = 1 if model_average > 0.7 else 0
    results.append(prediction)

  print("similarities[0]", similarities[0])
  print("prediction_confidence[0]", prediction_confidence[0])
  print("results[0]", results[0])
  print("labels[0]", labels[0])

  print("\nsimilarities[1]", similarities[1])
  print("prediction_confidence[1]", prediction_confidence[1])
  print("results[1]", results[1])
  print("labels[1]", labels[1])


  plot_confusion_matrix(labels, results, title='Complete Evaluation Confusion Matrix')
