import pickle
import matplotlib.pyplot as plt

# Load the .pkl file
with open('adasyn_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Extract loss and accuracy (error)
train_loss = history['validation_0']['mlogloss']
val_loss = history['validation_1']['mlogloss']
train_error = history['validation_0']['merror']
val_error = history['validation_1']['merror']

# Convert error to accuracy
train_accuracy = [1 - e for e in train_error]
val_accuracy = [1 - e for e in val_error]

# Plotting the loss and accuracy
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
