import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import pickle

# Memuat fitur dari file NPY
test_data = np.load('val.npy')

# Memisahkan fitur dan label dari data
test_features = test_data[:, :-1]
test_labels = test_data[:, -1].astype(int) # Mengonversi label menjadi tipe data int


# Assuming best_model is an instance of XGBClassifier already trained
model = joblib.load('xgb_adasyn_checkpoint.pkl')
test_predictions = model.predict(test_features)

# Calculate evaluation metrics
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

# Print the evaluation results
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-score:", test_f1)

# Print the classification report
print(classification_report(test_labels, test_predictions))
