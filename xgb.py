import numpy as np
# Memuat fitur dari file NPY
train_data = np.load('train.npy')

# Memisahkan fitur dan label dari data
train_features = train_data[:, :-1]
train_labels = train_data[:, -1].astype(int) # Mengonversi label menjadi tipe data int

import pandas as pd
# Create an initial DataFrame with columns
hasil_tuning_xgb = pd.DataFrame(columns=['param', 'accuracy', 'precision', 'recall', 'f1'])
# Save the initial DataFrame to a CSV file
hasil_tuning_xgb.to_csv('hasil_tuning_xgb.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from itertools import product
from tqdm import tqdm
import joblib
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Calculate the total number of features
total_features = train_features.shape[1]

# Create an empty list to store results
results_list = []



# Hyperparameter tuning XGBoost
param_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.2, 0.3],

}

param_list = list(map(
    lambda x: dict(zip(param_grid.keys(), x)),
    product(*param_grid.values())
))

# Initialize variables to keep track of the best model and its performance
best_acc = 0
best_param = None

for param in tqdm(param_list):
    model = xgb.XGBClassifier(n_jobs=9, random_state=42)
    model.set_params(**param)
    
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    result = {'param': str(param), 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    results_list.append(result)
    
    if accuracy > best_acc:
        best_acc = accuracy
        best_param = param
    
    # Save the model checkpoint
    joblib.dump(model, 'xgb_checkpoint.pkl')
    print('parameter: ', param)
    print('accuracy: ', accuracy)
    print('best param: ', best_param, ', best accuracy: ', best_acc, '\n')

# Convert the list of results to a DataFrame
hasil_tuning_xgb = pd.DataFrame(results_list)

# Save the DataFrame to a CSV file
hasil_tuning_xgb.to_csv('hasil_tuning_xgb.csv', index=False)

import matplotlib.pyplot as plt

# Load the DataFrame from the CSV file
hasil_tuning_xgb = pd.read_csv('hasil_tuning_xgb.csv')

# Sort the DataFrame based on the 'accuracy' column (you can choose a different column)
hasil_tuning_xgb = hasil_tuning_xgb.sort_values(by='accuracy', ascending=False)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(hasil_tuning_xgb['param'], hasil_tuning_xgb['accuracy'], color='skyblue')
plt.xlabel('Parameters')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Parameters')
plt.xticks(rotation=90) # Rotate the parameter labels for readability
plt.tight_layout()
plt.show()