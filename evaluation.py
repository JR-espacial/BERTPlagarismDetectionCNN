from Model import plot_confusion_matrix
import numpy as np

def complete_evaluation(model, data, labels, similarities):
  data = np.squeeze(data, axis=1)
  prediction_confidence = model.predict(data)
  results = []
  for i in range(len(data)):
    similarity_average = sum(similarities[i]) / len(similarities[i])
    model_average = (similarity_average + prediction_confidence[i]) / 2
    prediction = 1 if model_average > 0.6 else 0
    results.append(prediction)

  plot_confusion_matrix(labels, results, title='Complete Evaluation Confusion Matrix')
