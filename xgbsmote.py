import pandas as pd
hasil_tuning_rf_smote = pd.DataFrame(columns=['param', 'accuracy', 'precision', 'recall', 'f1'])
hasil_tuning_rf_smote.to_csv('tuning_xgb_smote.csv', index=False)

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from itertools import product
from tqdm import tqdm
from xgboost import XGBClassifier

# Ignore all warnings
warnings.filterwarnings("ignore")

# Memuat fitur dari file NPY
train_data = np.load('train.npy')

# Memisahkan fitur dan label dari data
train_features = train_data[:, :-1]
train_labels = train_data[:, -1].astype(int) # Mengonversi label menjadi tipe data int

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning XGBoost with SMOTE and k-fold cross validation

param_grid = {
'sampling_strategy': [ #oversampling the minority class
{0: 500, 1: 500, 3: 500},
{0: 1000, 1: 1000, 3: 1000},
],
'k_neighbors': [3, 5, 7]}

# 'model__n_estimators': [100, 200, 300],
# 'model__learning_rate': [0.01, 0.1, 0.2],


param_list = list(map(
lambda x: dict(zip(param_grid.keys(), x)),
product(*param_grid.values())
))

best_param, best_acc = {}, 0

hasil_tuning_xgb_smote = pd.read_csv('tuning_xgb_smote.csv')
for param in tqdm(param_list):
    smote = SMOTE(n_jobs=10) # Resampling menggunakan SMOTE
    smote.set_params(**param)

# Model klasifikasi - XGBoost
model_xgb = XGBClassifier(n_jobs=9, random_state=42) # Set n_jobs to -10 for parallel processing
best_param_xgb = {'n_estimators': 100, 'learning_rate': 0.3}
model_xgb.set_params(**best_param_xgb)

# SMOTE dan model klasifikasi
X, y = smote.fit_resample(train_features, train_labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

result = {'param': str(param), 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
hasil_tuning_xgb_smote = hasil_tuning_xgb_smote._append(result, ignore_index=True)
hasil_tuning_xgb_smote.to_csv('tuning_xgb_smote.csv', index=False)

if accuracy > best_acc:
    best_acc = accuracy
    best_param = param

# Save the model checkpoint
joblib.dump(model_xgb, 'xgb_smote_checkpoint.pkl')
print('parameter: ', param)
print('accuracy: ', accuracy)
print('best param: ', best_param, ', best accuracy: ', best_acc, '\n')

# Print best parameters and accuracy
print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_acc)

import matplotlib.pyplot as plt

# Load the tuning results from CSV
hasil_tuning_xgb_smote = pd.read_csv('tuning_xgb_smote.csv')

# Sort the DataFrame by accuracy in descending order
hasil_tuning_xgb_smote = hasil_tuning_xgb_smote.sort_values(by='accuracy', ascending=False)

# Create a bar plot for accuracy vs. parameter
fig, ax = plt.subplots(figsize=(10, 6))

# Extract parameter values and accuracy scores
param_values = [str(param) for param in hasil_tuning_xgb_smote['param']]
accuracy_scores = hasil_tuning_xgb_smote['accuracy']

# Plot the bar chart
ax.bar(param_values, accuracy_scores)
ax.set_xlabel('Parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs. Parameter (Sorted)')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90, ha="right")

plt.tight_layout()
plt.show()
