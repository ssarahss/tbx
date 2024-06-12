import csv

# Data from the table
data = [
    ["   ", "precision", "recall", "f1-score", "support"],
    [0, 0.00, 0.00, 0.00, 7],
    [1, 0.76, 0.63, 0.69, 157],
    [2, 0.94, 0.98, 0.96, 800],
    [3, 0.00, 0.00, 0.00, 36],
    [4, 0.94, 0.98, 0.96, 800],
    ["accuracy", "", 0.93, 1800],
    ["macro avg", 0.53, 0.52, 0.52, 1800],
    ["weighted avg", 0.90, 0.93, 0.91, 1800]
]

# Writing to CSV file
with open('rf.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
