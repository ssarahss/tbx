import pandas as pd
hasil_tuning_xgb_adasyn = pd.DataFrame(columns=['param', 'accuracy', 'precision', 'ecall', 'f1'])
hasil_tuning_xgb_adasyn.to_csv('tuning_xgb_adasyn.csv', index=False)

import warnings
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from itertools import product
from tqdm import tqdm
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle

# Ignore all warnings
warnings.filterwarnings("ignore")

# Memuat fitur dari file NPY
train_data = np.load('train.npy')

# Memisahkan fitur dan label dari data
train_features = train_data[:, :-1]
train_labels = train_data[:, -1].astype(int)  # Mengonversi label menjadi tipe data int

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Print the dataset before ADASYN
print("X_train before ADASYN:")
print(X_train)
print("y_train before ADASYN:")
print(y_train)

# Hyperparameter tuning XGBoost with ADASYN and k-fold cross validation



param_grid = {
    'sampling_strategy': [
        {0: 500, 1: 500, 3: 500},
        {0: 1000, 1: 1000, 3: 1000},
        {0: 1500, 1: 1500, 3: 1500}
    ],
    'n_neighbors': [3, 5, 7]
}

param_list = list(map(
    lambda x: dict(zip(param_grid.keys(), x)),
    product(*param_grid.values())
))

best_param, best_acc = {}, 0

hasil_tuning_xgb_adasyn = pd.read_csv('tuning_xgb_adasyn.csv')
for param in tqdm(param_list):
 
    adasyn = ADASYN(**param)  # Resampling menggunakan ADASYN

    # Check class distribution before resampling
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution before ADASYN: {class_distribution}")

    # Fit and transform the training data
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    # Check class distribution after resampling
    unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
    class_distribution_resampled = dict(zip(unique_resampled, counts_resampled))
    print(f"Class distribution after ADASYN: {class_distribution_resampled}")


    # Print the dataset after ADASYN
    print("X_train after ADASYN:")
    print(X_train_resampled)
    print("y_train after ADASYN:")
    print(y_train_resampled)

    # Model klasifikasi - XGBoost
    model_xgb = XGBClassifier(n_jobs=9, random_state=42)  # Set n_jobs to -10 for parallel processing
    best_param_xgb = {'n_estimators': 100, 'learning_rate': 0.3}
    model_xgb.set_params(**best_param_xgb)

    # Set up the evaluation sets and metrics
    eval_set = [(X_train_resampled, y_train_resampled), (X_val, y_val)]
    eval_metric = ['mlogloss', 'merror']

    # Train the model on the resampled data
    model_xgb.fit(X_train_resampled, y_train_resampled, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

    # Make predictions on the validation set
    y_pred = model_xgb.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Store the results
    hasil_tuning_xgb_adasyn.loc[len(hasil_tuning_xgb_adasyn)] = [str(param), accuracy, precision, recall, f1]

    # Check if this is the best model so far
    if accuracy > best_acc:
        best_param = param
        best_acc = accuracy
   
    # Save the model checkpoint
    joblib.dump(model_xgb, 'xgb_adasyn_checkpoint.pkl')
    print('parameter: ', param)
    print('accuracy: ', accuracy)
    print('best param: ', best_param, ', best accuracy: ', best_acc, '\n')

    # Save the evaluation results
    history = model_xgb.evals_result()
    with open('adasyn_history.pkl', 'wb') as f:
        pickle.dump(history, f)


hasil_tuning_xgb_adasyn.to_csv('tuning_xgb_adasyn.csv', index=False)

print("Best parameter:", best_param)
print("Best accuracy:", best_acc)


import matplotlib.pyplot as plt

# Load the tuning results from CSV
hasil_tuning_xgb_adasyn = pd.read_csv('tuning_xgb_adasyn.csv')

# Sort the DataFrame by accuracy in descending order
hasil_tuning_xgb_adasyn = hasil_tuning_xgb_adasyn.sort_values(by='accuracy', ascending=False)

# Create a bar plot for accuracy vs. parameter
fig, ax = plt.subplots(figsize=(10, 6))

# Extract parameter values and accuracy scores
param_values = [str(param) for param in hasil_tuning_xgb_adasyn['param']]
accuracy_scores = hasil_tuning_xgb_adasyn['accuracy']

# Plot the bar chart
ax.bar(param_values, accuracy_scores)
ax.set_xlabel('Parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs. Parameter (Sorted)')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90, ha="right")

plt.tight_layout()
plt.show()