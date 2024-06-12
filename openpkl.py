import pickle

# Loading the object from the .pkl file
with open('xgb_smote_checkpoint.pkl', 'rb') as file:
    loaded_object = pickle.load(file)

print(loaded_object)
