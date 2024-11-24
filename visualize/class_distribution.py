import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Extract predicted labels
predicted_labels = predictions['Label'].values

# Count the occurrences of each class
class_counts = np.bincount(predicted_labels)

# Plot bar chart of class distribution
plt.figure(figsize=(10, 6))
plt.bar(range(len(class_counts)), class_counts, color='green', edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
